"""
Evaluate MoE models with low-rank approximations of expert weights.

Loads a checkpoint, truncates expert weight matrices (fc1, fc2, temporal_conv)
to various ranks via SVD, and runs evaluation to measure performance retention.

Usage:
    python -m src.analysis.eval_lowrank_experts \
        --model_path checkpoints/.../ihm_TS-Text_mod_drop_rate_0.0.pt \
        --encoder_path checkpoints/.../ihm_TS-Text_mod_drop_rate_0.0_IHM_mod_drop_rate_0.0_encoder.pt \
        --task ihm --ihm_mod TS-Text --los_mod TS-CXR --pheno_mod TS-Text-CXR \
        --ranks 1 2 4 8 16 32 64 full \
        --output_dir analysis/analysis_results/lowrank_eval
        <other args matching the training config>
"""

import sys
import os
sys.path.insert(1, os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score

from src.eval_scripts.performance import metrics_multilabel, metrics_multiclass
from src.train_structure_multitask_mimic import drop_modalities, replace_missing_embeddings
from src.mimiciv_tasks import parse_args, loadBert
from src.mimiciv_task_setup import setup_tasks_and_modalities
from src.crossattnperceiver import InputModality
from src.fusemoe import MULTCrossModel
from transformers import set_seed
from accelerate import Accelerator

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def truncate_to_rank(W, rank):
    """Apply rank-k SVD approximation to a 2D weight matrix. Rank 0 returns zeros."""
    if rank == 0:
        return torch.zeros_like(W)
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    k = min(rank, len(S))
    W_approx = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]
    return W_approx.to(W.dtype)


def apply_lowrank_to_experts(model, rank):
    """Replace all expert weight matrices with their rank-k SVD approximation.

    Modifies the model in-place. Handles:
      - fc1.weight, fc2.weight: 2D [out, in]
      - temporal_conv.weight: 3D [out, in, kernel] -> reshape to 2D, truncate, reshape back
    """
    sd = model.state_dict()
    modified = 0

    for key in sd:
        if '.moe.experts.' not in key:
            continue
        if not key.endswith('.weight'):
            continue
        # Skip LayerNorm weights (1D)
        if sd[key].dim() == 1:
            continue

        W = sd[key]
        if W.dim() == 2:
            sd[key] = truncate_to_rank(W, rank)
            modified += 1
        elif W.dim() == 3:
            # Conv1d: [out_channels, in_channels, kernel_size]
            out_c, in_c, ks = W.shape
            W_2d = W.reshape(out_c, in_c * ks)
            W_2d_approx = truncate_to_rank(W_2d, rank)
            sd[key] = W_2d_approx.reshape(out_c, in_c, ks)
            modified += 1

    model.load_state_dict(sd)
    return modified


def diagnose_expert_contribution(model, encoder, test, modalities, args, device, max_batches=10):
    """Comprehensive diagnosis of expert contribution and MoE-residual redundancy.

    Measures:
    1. Per-expert output norms and input/output ratios
    2. MoE combined output statistics
    3. Cosine similarity between residual and MoE output (redundancy)
    4. Residual projection ratio (fraction of MoE output in residual direction)
    5. Cross-expert output similarity (do experts specialize?)
    6. Effective rank of MoE output across samples
    7. Unique information in MoE output (component orthogonal to residual)
    """
    model.eval()
    for enc in encoder.values():
        enc.eval()

    # Hook storage
    expert_stats = {}
    moe_io_stats = []

    # Per-layer residual vs MoE output storage (read from _diag_* attrs)
    layer_diag = {}  # layer_name -> list of per-batch dicts

    # Hook experts
    expert_hooks = []
    for name, module in model.named_modules():
        if 'TemporalExpertMLP' in type(module).__name__ or 'FlexExpertMLP' in type(module).__name__:
            stats_list = []
            expert_stats[name] = stats_list
            def make_hook(sl):
                def hook_fn(mod, inp, out):
                    with torch.no_grad():
                        sl.append({
                            'input_norm': inp[0].float().norm().item(),
                            'output_norm': out.float().norm().item(),
                            'output_mean': out.float().mean().item(),
                            'output_std': out.float().std().item(),
                            'output_abs_mean': out.float().abs().mean().item(),
                            'input_abs_mean': inp[0].float().abs().mean().item(),
                        })
                return hook_fn
            expert_hooks.append(module.register_forward_hook(make_hook(stats_list)))

    # Hook MoE modules to capture combined output
    moe_hooks = []
    for name, module in model.named_modules():
        if 'SeqMoE' in type(module).__name__ or 'FlexSeqMoE' in type(module).__name__:
            def make_moe_hook(sl):
                def hook_fn(mod, inp, out):
                    with torch.no_grad():
                        if isinstance(out, tuple) and out[0] is not None:
                            out_list = out[0]
                            for o in out_list:
                                sl.append({
                                    'moe_output_norm': o.float().norm().item(),
                                    'moe_output_abs_mean': o.float().abs().mean().item(),
                                    'moe_output_std': o.float().std().item(),
                                })
                return hook_fn
            moe_hooks.append(module.register_forward_hook(make_moe_hook(moe_io_stats)))

    # Hook TransformerCrossEncoderLayers to read _diag_residual / _diag_moe_output
    layer_hooks = []
    for name, module in model.named_modules():
        if 'TransformerCrossEncoderLayer' in type(module).__name__:
            diag_list = []
            layer_diag[name] = diag_list
            def make_layer_hook(nm, dl):
                def hook_fn(mod, inp, out):
                    with torch.no_grad():
                        if hasattr(mod, '_diag_residual') and hasattr(mod, '_diag_moe_output'):
                            for res, moe_out in zip(mod._diag_residual, mod._diag_moe_output):
                                r = res.float()
                                m = moe_out.float()

                                # Flatten to [num_tokens, D]
                                r_flat = r.reshape(-1, r.shape[-1])
                                m_flat = m.reshape(-1, m.shape[-1])

                                # 1. Cosine similarity (per-token, then average)
                                cos_sim = F.cosine_similarity(r_flat, m_flat, dim=-1)
                                avg_cos = cos_sim.mean().item()

                                # 2. Residual projection ratio:
                                #    ||proj_r(m)|| / ||m|| = how much of MoE lies along residual
                                r_norm = r_flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
                                r_unit = r_flat / r_norm
                                proj_scalar = (m_flat * r_unit).sum(dim=-1, keepdim=True)
                                proj_onto_r = proj_scalar * r_unit
                                proj_ratio = proj_onto_r.norm(dim=-1) / m_flat.norm(dim=-1).clamp(min=1e-10)
                                avg_proj_ratio = proj_ratio.mean().item()

                                # 3. Orthogonal component: m - proj_r(m)
                                orth = m_flat - proj_onto_r
                                orth_norm = orth.norm(dim=-1).mean().item()
                                m_norm_avg = m_flat.norm(dim=-1).mean().item()
                                orth_frac = orth_norm / m_norm_avg if m_norm_avg > 1e-10 else 0.0

                                # 4. Effective rank of MoE output (via SVD on token matrix)
                                if m_flat.shape[0] > 1:
                                    S = torch.linalg.svdvals(m_flat)
                                    S = S[S > 1e-8]
                                    if len(S) > 0:
                                        p = S / S.sum()
                                        eff_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-10))).item()
                                    else:
                                        eff_rank = 0.0
                                else:
                                    eff_rank = 0.0

                                alpha = mod._diag_alpha if hasattr(mod, '_diag_alpha') else None

                                dl.append({
                                    'cos_sim': avg_cos,
                                    'proj_ratio': avg_proj_ratio,
                                    'orth_frac': orth_frac,
                                    'orth_norm': orth_norm,
                                    'moe_norm': m_norm_avg,
                                    'residual_norm': r_flat.norm(dim=-1).mean().item(),
                                    'eff_rank': eff_rank,
                                    'max_rank': min(m_flat.shape),
                                    'alpha': alpha,
                                })
                return hook_fn
            layer_hooks.append(module.register_forward_hook(make_layer_hook(name, diag_list)))

    # Collect per-expert outputs per layer for cross-expert similarity
    expert_output_hooks = []
    expert_outputs_per_layer = {}  # layer_prefix -> expert_idx -> list of flattened outputs
    for name, module in model.named_modules():
        if 'TemporalExpertMLP' in type(module).__name__ or 'FlexExpertMLP' in type(module).__name__:
            # Parse layer and expert index from name like
            # trans_self_cross_ts_txt.layers.0.moe.experts.1
            parts = name.split('.')
            # Find "experts" and get index
            for pi, p in enumerate(parts):
                if p == 'experts' and pi + 1 < len(parts):
                    layer_prefix = '.'.join(parts[:pi])
                    expert_idx = int(parts[pi + 1])
                    if layer_prefix not in expert_outputs_per_layer:
                        expert_outputs_per_layer[layer_prefix] = {}
                    if expert_idx not in expert_outputs_per_layer[layer_prefix]:
                        expert_outputs_per_layer[layer_prefix][expert_idx] = []
                    out_list = expert_outputs_per_layer[layer_prefix][expert_idx]
                    def make_expert_out_hook(ol):
                        def hook_fn(mod, inp, out):
                            with torch.no_grad():
                                ol.append(out.float().reshape(-1).cpu())
                        return hook_fn
                    expert_output_hooks.append(module.register_forward_hook(make_expert_out_hook(out_list)))
                    break

    # Run batches
    task_names = {'MOR': 'mortality', 'RAD': 'readmission'}
    with torch.no_grad():
        for ii in range(len(test)):
            task = modalities[int(ii)][0].split('_')[1]
            model.to_logits = model.to_logitslist[ii]
            batch_count = 0

            for jj in test[ii]:
                if batch_count >= max_batches:
                    break

                if task in ['IHM', 'PHENO', 'LOS']:
                    ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts, \
                        input_ids_sequences, attn_mask_sequences, text_emb, \
                        note_time, note_time_mask, cxr_feats, cxr_time, \
                        cxr_time_mask, ecg_feats, ecg_time, ecg_time_mask, \
                        label, cxr_missing, text_missing, ecg_missing = jj
                    data = {
                        'ts_input_sequences': ts_input_sequences, 'ts_mask_sequences': ts_mask_sequences,
                        'ts_tt': ts_tt, 'reg_ts': reg_ts,
                        'input_ids_sequences': input_ids_sequences, 'attn_mask_sequences': attn_mask_sequences,
                        'text_emb': text_emb, 'note_time': note_time, 'note_time_mask': note_time_mask,
                        'cxr_feats': cxr_feats, 'cxr_time': cxr_time, 'cxr_time_mask': cxr_time_mask,
                        'ecg_feats': ecg_feats, 'ecg_time': ecg_time, 'ecg_time_mask': ecg_time_mask,
                        'label': label, 'cxr_missing': cxr_missing,
                        'text_missing': text_missing, 'ecg_missing': ecg_missing,
                    }
                elif task in ['BIRADS', 'RISK', 'DENSITY']:
                    idx, label, embed_2dcc, embed_2dmlo, embed_cc, embed_mlo, all_views = jj
                    data = {
                        'embed_cc': embed_cc, 'embed_mlo': embed_mlo,
                        'embed_2dcc': embed_2dcc, 'embed_2dmlo': embed_2dmlo,
                        'all_views': all_views, 'label': label,
                    }
                elif task in ['MOR', 'RAD']:
                    data = {
                        'codes': jj['codes'], 'types': jj['types'],
                        'timestamps': jj['timestamps'], 'ages': jj['age'],
                        'genders': jj['gender'], 'ethnicities': jj['ethnicity'],
                        'label': jj[task_names[task]].long(),
                    }
                else:
                    continue

                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(device)

                try:
                    if task in ['IHM', 'PHENO', 'LOS']:
                        encoded = encoder[task](
                            x_ts=data['ts_input_sequences'],
                            x_ts_mask=data['ts_mask_sequences'],
                            ts_tt_list=data['ts_tt'],
                            input_ids_sequences=data['input_ids_sequences'],
                            attn_mask_sequences=data['attn_mask_sequences'],
                            text_emb=data['text_emb'],
                            note_time_list=data['note_time'],
                            note_time_mask_list=data['note_time_mask'],
                            cxr_feats=data['cxr_feats'],
                            cxr_time=data['cxr_time'],
                            cxr_time_mask=data['cxr_time_mask'],
                            ecg_feats=data['ecg_feats'],
                            ecg_time=data['ecg_time'],
                            ecg_time_mask=data['ecg_time_mask'],
                            labels=data['label'], reg_ts=data['reg_ts'],
                            cxr_missing=data['cxr_missing'],
                            text_missing=data['text_missing'],
                            ecg_missing=data['ecg_missing'],
                            modalities=modalities[ii],
                        )
                    elif task in ['BIRADS', 'RISK', 'DENSITY']:
                        encoded = encoder[task](
                            embed_cc=data['embed_cc'], embed_mlo=data['embed_mlo'],
                            embed_2dcc=data['embed_2dcc'], embed_2dmlo=data['embed_2dmlo'],
                            all_views=data['all_views'],
                            modalities=modalities[ii], task=task,
                        )
                    elif task in ['MOR', 'RAD']:
                        encoded = encoder[task](
                            codes=data['codes'], types=data['types'],
                            timestamps=data['timestamps'], ages=data['ages'],
                            genders=data['genders'], ethnicities=data['ethnicities'],
                            modalities=modalities[int(ii)],
                        )

                    indict = {}
                    for i in range(len(modalities[ii])):
                        indict[modalities[ii][i]] = encoded[modalities[ii][i]].float().to(device)
                    model(indict, task=task)
                except Exception as e:
                    print(f"  Warning: batch failed in {task}: {e}")
                    continue

                batch_count += 1

    # Remove all hooks
    for h in expert_hooks + moe_hooks + layer_hooks + expert_output_hooks:
        h.remove()

    # ======================== PRINT RESULTS ========================
    print(f"\n{'='*60}")
    print("  Expert Contribution Diagnosis")
    print(f"{'='*60}")

    # --- 1. Per-Expert Output Statistics ---
    print("\n  Per-Expert Output Statistics (averaged over batches):")
    for name, stats in expert_stats.items():
        if not stats:
            print(f"    {name}: NO ACTIVATIONS (never called)")
            continue
        avg_in_norm = np.mean([s['input_norm'] for s in stats])
        avg_out_norm = np.mean([s['output_norm'] for s in stats])
        avg_out_abs = np.mean([s['output_abs_mean'] for s in stats])
        avg_in_abs = np.mean([s['input_abs_mean'] for s in stats])
        avg_out_std = np.mean([s['output_std'] for s in stats])
        ratio = avg_out_norm / avg_in_norm if avg_in_norm > 1e-10 else float('inf')
        print(f"    {name} ({len(stats)} calls):")
        print(f"      input:  norm={avg_in_norm:.4f}, abs_mean={avg_in_abs:.6f}")
        print(f"      output: norm={avg_out_norm:.4f}, abs_mean={avg_out_abs:.6f}, std={avg_out_std:.6f}")
        print(f"      output/input norm ratio: {ratio:.4f}")

    # --- 2. MoE Combined Output Statistics ---
    print("\n  MoE Combined Output Statistics:")
    if moe_io_stats:
        avg_moe_norm = np.mean([s['moe_output_norm'] for s in moe_io_stats])
        avg_moe_abs = np.mean([s['moe_output_abs_mean'] for s in moe_io_stats])
        avg_moe_std = np.mean([s['moe_output_std'] for s in moe_io_stats])
        print(f"    moe_output: norm={avg_moe_norm:.4f}, abs_mean={avg_moe_abs:.6f}, std={avg_moe_std:.6f}")
    else:
        print(f"    No MoE output captured")

    # --- 3. MoE vs Residual Redundancy Analysis (per layer) ---
    print(f"\n{'='*60}")
    print("  MoE vs Residual Redundancy Analysis")
    print(f"{'='*60}")

    for layer_name, diag_list in layer_diag.items():
        if not diag_list:
            continue
        print(f"\n  {layer_name}:")
        avg_cos = np.mean([d['cos_sim'] for d in diag_list])
        avg_proj = np.mean([d['proj_ratio'] for d in diag_list])
        avg_orth_frac = np.mean([d['orth_frac'] for d in diag_list])
        avg_orth_norm = np.mean([d['orth_norm'] for d in diag_list])
        avg_moe_norm = np.mean([d['moe_norm'] for d in diag_list])
        avg_res_norm = np.mean([d['residual_norm'] for d in diag_list])
        avg_eff_rank = np.mean([d['eff_rank'] for d in diag_list])
        max_rank = diag_list[0]['max_rank'] if diag_list else 0
        alpha = diag_list[0]['alpha']

        if alpha is not None:
            print(f"    alpha (residual weight): {alpha:.4f}, MoE weight: {1-alpha:.4f}")
        print(f"    residual norm: {avg_res_norm:.4f}")
        print(f"    MoE output norm: {avg_moe_norm:.4f}")
        print(f"    cosine similarity (residual, MoE output): {avg_cos:.4f}")
        print(f"      (1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite)")
        print(f"    projection ratio ||proj_res(MoE)|| / ||MoE||: {avg_proj:.4f}")
        print(f"      (1.0 = MoE output fully in residual direction, 0.0 = fully orthogonal)")
        print(f"    orthogonal fraction ||MoE - proj|| / ||MoE||: {avg_orth_frac:.4f}")
        print(f"      (unique information not in residual: {avg_orth_frac:.2%})")
        print(f"    orthogonal component norm: {avg_orth_norm:.4f}")
        print(f"    MoE output effective rank: {avg_eff_rank:.1f} / {max_rank}")

    # --- 4. Cross-Expert Similarity ---
    print(f"\n{'='*60}")
    print("  Cross-Expert Similarity (do experts specialize?)")
    print(f"{'='*60}")

    for layer_prefix, experts_dict in expert_outputs_per_layer.items():
        expert_idxs = sorted(experts_dict.keys())
        if len(expert_idxs) < 2:
            continue
        print(f"\n  {layer_prefix}:")
        # Compute average output per expert (across all batches)
        expert_means = {}
        for eidx in expert_idxs:
            outs = experts_dict[eidx]
            nonempty = [o for o in outs if len(o) > 0]
            if nonempty:
                # Outputs vary in size across batches; truncate to common min length
                min_size = min(len(o) for o in nonempty)
                if min_size > 0:
                    expert_means[eidx] = torch.stack([o[:min_size] for o in nonempty]).mean(dim=0)

        # Pairwise cosine similarity between expert mean outputs
        for i in range(len(expert_idxs)):
            for j in range(i + 1, len(expert_idxs)):
                ei, ej = expert_idxs[i], expert_idxs[j]
                if ei in expert_means and ej in expert_means:
                    min_len = min(len(expert_means[ei]), len(expert_means[ej]))
                    cos = F.cosine_similarity(
                        expert_means[ei][:min_len].unsqueeze(0),
                        expert_means[ej][:min_len].unsqueeze(0)
                    ).item()
                    print(f"    expert {ei} vs expert {ej}: cosine_sim = {cos:.4f}")

    # --- 5. Summary interpretation ---
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")

    all_cos = []
    all_orth = []
    for dl in layer_diag.values():
        all_cos.extend([d['cos_sim'] for d in dl])
        all_orth.extend([d['orth_frac'] for d in dl])

    if all_cos:
        grand_cos = np.mean(all_cos)
        grand_orth = np.mean(all_orth)
        print(f"\n    Avg cosine sim (residual, MoE): {grand_cos:.4f}")
        print(f"    Avg orthogonal fraction: {grand_orth:.4f}")

        if grand_cos > 0.9:
            print("\n    CONCLUSION: MoE output is HIGHLY REDUNDANT with residual.")
            print("    MoE has learned to approximate a scaled version of the residual.")
            print("    Rank truncation has no effect because the information is duplicated.")
        elif grand_cos > 0.5:
            print("\n    CONCLUSION: MoE output is PARTIALLY REDUNDANT with residual.")
            print(f"    ~{(1-grand_orth)*100:.0f}% of MoE output overlaps with residual direction.")
            print(f"    ~{grand_orth*100:.0f}% is unique information not in the residual.")
        elif grand_cos > -0.1:
            print("\n    CONCLUSION: MoE output is LARGELY INDEPENDENT of residual.")
            print("    MoE contributes distinct information, but rank invariance")
            print("    suggests this information is low-dimensional or task-irrelevant.")
        else:
            print("\n    CONCLUSION: MoE output OPPOSES the residual (negative cosine).")
            print("    MoE may be learning a subtractive correction.")

    return expert_stats, moe_io_stats, layer_diag


def count_expert_params(model, rank=None):
    """Count parameters in expert weight matrices.

    For full rank: count all elements in expert weight/bias tensors.
    For low rank k: a rank-k matrix [m, n] is stored as U[m,k] + S[k] + Vh[k,n] = k*(m+n+1).
    Bias params are always counted at full size.

    Returns:
        full_params: total expert params at full rank
        lowrank_params: total expert params if stored in factored form at given rank
        details: per-weight breakdown
    """
    sd = model.state_dict()
    full_params = 0
    lowrank_params = 0
    details = []

    for key in sorted(sd):
        if '.moe.experts.' not in key:
            continue

        W = sd[key]
        numel = W.numel()
        full_params += numel

        if rank is None or W.dim() < 2 or not key.endswith('.weight'):
            lowrank_params += numel
        else:
            if W.dim() == 2:
                m, n = W.shape
                k = min(rank, min(m, n))
                factored = k * (m + n + 1)
                lowrank_params += factored
                details.append((key, f'{list(W.shape)}', numel, factored))
            elif W.dim() == 3:
                out_c, in_c, ks = W.shape
                m, n = out_c, in_c * ks
                k = min(rank, min(m, n))
                factored = k * (m + n + 1)
                lowrank_params += factored
                details.append((key, f'{list(W.shape)}', numel, factored))
            else:
                lowrank_params += numel

    return full_params, lowrank_params, details


def derive_run_subdir(model_path):
    """Extract the run-identifying sub-path from a checkpoint path.

    Mirrors the directory structure under `checkpoints/` so analysis output
    lands at the same relative location as the source checkpoint, e.g.
        .../checkpoints/flame_w_balanced_loss_1.0/.../birads_cc-mlo_lr1e-4/foo.pt
        -> flame_w_balanced_loss_1.0/.../birads_cc-mlo_lr1e-4

    Falls back to the immediate parent directory name if `checkpoints` is not
    in the path components.
    """
    parent = os.path.dirname(os.path.abspath(model_path))
    parts = parent.split(os.sep)
    if 'checkpoints' in parts:
        idx = parts.index('checkpoints')
        tail = parts[idx + 1:]
        if tail:
            return os.path.join(*tail)
    return os.path.basename(parent)


def categorize_layer(name):
    """Bucket a state_dict key into a module category by walking its dotted path.

    Specific module types (experts, router, attention) take precedence; anything
    that doesn't match a keyword falls back to the top-level module name so
    every layer ends up in some plot rather than getting silently dropped.
    """
    parts = name.split('.')
    parts_set = set(parts)

    if 'experts' in parts_set:
        return 'experts'
    if parts_set & {'w_gate', 'w_noise', 'gate', 'routers', 'router'}:
        return 'router'
    if 'cross_attn' in parts_set or 'crossattn' in name.lower():
        return 'cross_attention'
    if 'self_attn' in parts_set or 'selfattn' in name.lower():
        return 'self_attention'
    if parts_set & {'to_logits', 'to_logitslist'}:
        return 'output_head'
    if any('embed' in p.lower() for p in parts):
        return 'embedding'
    if parts_set & {'linear1', 'linear2', 'fc1', 'fc2', 'mlp', 'ffn'}:
        return 'ffn'
    return parts[0] if parts else 'other'


def _trim_common_prefix(names):
    """Strip the longest dotted common prefix shared by all names, for cleaner labels."""
    if len(names) <= 1:
        return names
    prefix = os.path.commonprefix(names)
    if '.' in prefix:
        prefix = prefix.rsplit('.', 1)[0] + '.'
        return [n[len(prefix):] for n in names]
    return names


def plot_singular_value_spectrum(model, output_dir):
    """Plot SV spectrum heatmaps, one per detected module category.

    For each 2D weight matrix (Conv weights [out,in,k] are flattened to
    [out, in*k] first), compute singular values and group by `categorize_layer`.
    Each heatmap: rows = layers in that category, cols = SV index up to the
    max rank in that group, NaN-padded for layers with a smaller max rank.
    Cell color = singular value (log scale).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("matplotlib not available, skipping SV spectrum plot")
        return

    sd = model.state_dict()
    layers_by_category = {}

    for key in sorted(sd):
        if not key.endswith('.weight'):
            continue
        W = sd[key]
        if W.dim() < 2:
            continue
        if W.dim() == 3:
            out_c, in_c, ks = W.shape
            W2d = W.reshape(out_c, in_c * ks)
        elif W.dim() == 2:
            W2d = W
        else:
            continue
        try:
            S = torch.linalg.svdvals(W2d.float().cpu()).numpy()
        except Exception as e:
            print(f"  SVD failed for {key}: {e}")
            continue
        layers_by_category.setdefault(categorize_layer(key), []).append((key, S))

    if not layers_by_category:
        print("No 2D weight matrices found for SV spectrum.")
        return

    all_sv_data = {}
    for category in sorted(layers_by_category):
        layers = layers_by_category[category]
        max_rank = max(len(s) for _, s in layers)
        n_layers = len(layers)
        matrix = np.full((n_layers, max_rank), np.nan)
        for i, (_, s) in enumerate(layers):
            matrix[i, :len(s)] = s

        finite = matrix[np.isfinite(matrix) & (matrix > 0)]
        vmin = max(float(finite.min()), 1e-8) if finite.size > 0 else 1e-8
        vmax = float(finite.max()) if finite.size > 0 else 1.0

        fig_height = max(3.0, min(n_layers * 0.22, 30.0))
        fig, ax = plt.subplots(figsize=(12, fig_height))
        im = ax.imshow(matrix, aspect='auto', cmap='viridis',
                       norm=LogNorm(vmin=vmin, vmax=vmax),
                       interpolation='nearest')
        ax.set_xlabel(f'Singular-value index (max rank = {max_rank})')
        ax.set_ylabel('Layer')
        ax.set_title(f'SV Spectrum: {category} ({n_layers} layers)')
        ax.set_yticks(range(n_layers))
        short_names = _trim_common_prefix([n for n, _ in layers])
        ax.set_yticklabels(short_names, fontsize=6)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Singular value (log scale)')
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'singular_value_spectrum_{category}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  SV spectrum [{category}] ({n_layers} layers) saved: {save_path}")

        all_sv_data[category] = {n: s.tolist() for n, s in layers}

    json_path = os.path.join(output_dir, 'singular_value_spectrum.json')
    with open(json_path, 'w') as f:
        json.dump(all_sv_data, f, indent=2)
    print(f"  SV spectrum data saved: {json_path}")


def plot_expert_energy_curves(model, output_dir):
    """Plot cumulative spectral energy vs rank for every expert weight matrix.

    For each expert layer, computes E(k) = 100 * sum(S[:k]**2) / sum(S**2)
    where S are singular values sorted descending. The curve answers: "what
    fraction of this expert's spectral energy does a rank-k truncation
    capture?". Conv weights [out, in, ks] are flattened to [out, in*ks] first.
    Reference lines mark 90% and 99% energy thresholds.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping expert energy curves")
        return

    sd = model.state_dict()
    curves = []  # (name, cumulative_energy_pct)

    for key in sorted(sd):
        if not key.endswith('.weight'):
            continue
        if '.moe.experts.' not in key:
            continue
        W = sd[key]
        if W.dim() < 2:
            continue
        if W.dim() == 3:
            out_c, in_c, ks = W.shape
            W2d = W.reshape(out_c, in_c * ks)
        elif W.dim() == 2:
            W2d = W
        else:
            continue
        try:
            S = torch.linalg.svdvals(W2d.float().cpu()).numpy()
        except Exception as e:
            print(f"  SVD failed for {key}: {e}")
            continue
        sq = S ** 2
        total = float(sq.sum())
        if total <= 0:
            continue
        # Prepend 0 so the curve starts at (k=0, energy=0%) and ends at
        # (k=len(S), energy=100%).
        cum = np.concatenate(([0.0], np.cumsum(sq))) / total * 100.0
        curves.append((key, cum))

    if not curves:
        print("No expert weight matrices found for energy curves.")
        return

    short_names = _trim_common_prefix([n for n, _ in curves])
    n = len(curves)
    show_per_curve_labels = n <= 30

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('viridis')
    for i, ((_, cum), short) in enumerate(zip(curves, short_names)):
        color = cmap(i / max(1, n - 1)) if n > 1 else cmap(0.5)
        ax.plot(range(len(cum)), cum, color=color, alpha=0.6, linewidth=1.0,
                label=short if show_per_curve_labels else None)

    ax.axhline(90, color='red', linestyle='--', alpha=0.5, label='90% energy')
    ax.axhline(99, color='red', linestyle=':', alpha=0.5, label='99% energy')

    ax.set_xlabel('Rank K')
    ax.set_ylabel('% Cumulative Energy in Top-K Singular Values')
    ax.set_title(f'Expert Spectral Energy vs Rank ({n} layers)')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6 if show_per_curve_labels else 9,
              loc='lower right',
              ncol=2 if show_per_curve_labels else 1)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'expert_energy_curves.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Expert energy curves saved: {save_path}")

    json_path = os.path.join(output_dir, 'expert_energy_curves.json')
    with open(json_path, 'w') as f:
        json.dump({name: cum.tolist() for name, cum in curves}, f, indent=2)
    print(f"  Expert energy data saved: {json_path}")


def _run_test_forward_passes(model, encoder, all_test, modalities_per_task,
                              args, device, max_batches=10, on_task_start=None):
    """Drive forward passes over the test set, one task at a time.

    Pure side-effect: used by hook-based analyses (data-aware energy,
    routing distributions) to populate accumulators. Optional
    `on_task_start(task)` callback fires once per task before its batches run,
    so callers can stamp the current task into hook state.
    """
    model.eval()
    for enc in encoder.values():
        enc.eval()

    task_names = {'MOR': 'mortality', 'RAD': 'readmission'}
    with torch.no_grad():
        for ii in range(len(all_test)):
            task = modalities_per_task[int(ii)][0].split('_')[1]
            if on_task_start is not None:
                on_task_start(task)
            model.to_logits = model.to_logitslist[ii]
            batch_count = 0
            for jj in all_test[ii]:
                if batch_count >= max_batches:
                    break
                try:
                    if task in ['IHM', 'PHENO', 'LOS']:
                        (ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts,
                         input_ids_sequences, attn_mask_sequences, text_emb,
                         note_time, note_time_mask, cxr_feats, cxr_time,
                         cxr_time_mask, ecg_feats, ecg_time, ecg_time_mask,
                         label, cxr_missing, text_missing, ecg_missing) = jj
                        data = {
                            'ts_input_sequences': ts_input_sequences, 'ts_mask_sequences': ts_mask_sequences,
                            'ts_tt': ts_tt, 'reg_ts': reg_ts,
                            'input_ids_sequences': input_ids_sequences, 'attn_mask_sequences': attn_mask_sequences,
                            'text_emb': text_emb, 'note_time': note_time, 'note_time_mask': note_time_mask,
                            'cxr_feats': cxr_feats, 'cxr_time': cxr_time, 'cxr_time_mask': cxr_time_mask,
                            'ecg_feats': ecg_feats, 'ecg_time': ecg_time, 'ecg_time_mask': ecg_time_mask,
                            'label': label, 'cxr_missing': cxr_missing,
                            'text_missing': text_missing, 'ecg_missing': ecg_missing,
                        }
                    elif task in ['BIRADS', 'RISK', 'DENSITY']:
                        idx, label, embed_2dcc, embed_2dmlo, embed_cc, embed_mlo, all_views = jj
                        data = {
                            'embed_cc': embed_cc, 'embed_mlo': embed_mlo,
                            'embed_2dcc': embed_2dcc, 'embed_2dmlo': embed_2dmlo,
                            'all_views': all_views, 'label': label,
                        }
                    elif task in ['MOR', 'RAD']:
                        data = {
                            'codes': jj['codes'], 'types': jj['types'],
                            'timestamps': jj['timestamps'], 'ages': jj['age'],
                            'genders': jj['gender'], 'ethnicities': jj['ethnicity'],
                            'label': jj[task_names[task]].long(),
                        }
                    else:
                        continue

                    for k_, v in data.items():
                        if isinstance(v, torch.Tensor):
                            data[k_] = v.to(device)

                    if task in ['IHM', 'PHENO', 'LOS']:
                        encoded = encoder[task](
                            x_ts=data['ts_input_sequences'], x_ts_mask=data['ts_mask_sequences'],
                            ts_tt_list=data['ts_tt'],
                            input_ids_sequences=data['input_ids_sequences'],
                            attn_mask_sequences=data['attn_mask_sequences'],
                            text_emb=data['text_emb'],
                            note_time_list=data['note_time'], note_time_mask_list=data['note_time_mask'],
                            cxr_feats=data['cxr_feats'], cxr_time=data['cxr_time'], cxr_time_mask=data['cxr_time_mask'],
                            ecg_feats=data['ecg_feats'], ecg_time=data['ecg_time'], ecg_time_mask=data['ecg_time_mask'],
                            labels=data['label'], reg_ts=data['reg_ts'],
                            cxr_missing=data['cxr_missing'], text_missing=data['text_missing'],
                            ecg_missing=data['ecg_missing'],
                            modalities=modalities_per_task[ii],
                        )
                    elif task in ['BIRADS', 'RISK', 'DENSITY']:
                        encoded = encoder[task](
                            embed_cc=data['embed_cc'], embed_mlo=data['embed_mlo'],
                            embed_2dcc=data['embed_2dcc'], embed_2dmlo=data['embed_2dmlo'],
                            all_views=data['all_views'],
                            modalities=modalities_per_task[ii], task=task,
                        )
                    elif task in ['MOR', 'RAD']:
                        encoded = encoder[task](
                            codes=data['codes'], types=data['types'],
                            timestamps=data['timestamps'], ages=data['ages'],
                            genders=data['genders'], ethnicities=data['ethnicities'],
                            modalities=modalities_per_task[int(ii)],
                        )

                    indict = {}
                    for i in range(len(modalities_per_task[ii])):
                        indict[modalities_per_task[ii][i]] = encoded[modalities_per_task[ii][i]].float().to(device)
                    model(indict, task=task)
                except Exception as e:
                    print(f"  Warning: batch failed in {task}: {e}")
                    continue

                batch_count += 1


def plot_data_aware_energy_curves(model, encoder, all_test, modalities_per_task,
                                   args, device, output_dir, max_batches=5):
    """Cumulative spectral energy of expert weights, weighted by *real test
    activations* — reveals which ranks the network actually uses.

    For every nn.Linear inside an expert, accumulate the input second moment
    `Cx = E[xxᵀ]` over `max_batches` test forward passes, then for `W = UΣVᵀ`:

        E_data(k)   = Σᵢ≤k σᵢ² · vᵢᵀ Cx vᵢ  /  Σᵢ σᵢ² · vᵢᵀ Cx vᵢ
        E_weight(k) = Σᵢ≤k σᵢ²              /  Σᵢ σᵢ²

    The weight-only curve treats every direction equally (Frobenius energy);
    the data-aware curve weights direction i by how much the test data excites
    it, so it captures `‖Wx‖` energy on the actual distribution. The gap
    between the two curves is exactly the "functional rank" intuition.

    Conv weights are skipped (would need im2col-style unfold to map cleanly).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping data-aware energy plot")
        return

    # --- Install input-side covariance hooks on every expert Linear ---
    cx_accum = {}   # name -> [in_dim, in_dim] running sum of xᵀx
    n_samples = {}  # name -> total tokens seen
    weights = {}    # name -> W tensor (cpu, fp32)
    hooks = []

    for name, module in model.named_modules():
        if '.moe.experts.' not in name:
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        weights[name] = module.weight.detach().float().cpu()
        cx_accum[name] = None
        n_samples[name] = 0

        def make_hook(nm):
            def hook_fn(mod, inp, out):
                with torch.no_grad():
                    x = inp[0].float().detach()
                    x = x.reshape(-1, x.shape[-1])
                    xtx = (x.T @ x).cpu()
                    if cx_accum[nm] is None:
                        cx_accum[nm] = xtx
                    else:
                        cx_accum[nm] += xtx
                    n_samples[nm] += x.shape[0]
            return hook_fn
        hooks.append(module.register_forward_hook(make_hook(name)))

    if not weights:
        print("No expert Linear layers found for data-aware energy.")
        return

    _run_test_forward_passes(
        model, encoder, all_test, modalities_per_task,
        args, device, max_batches=max_batches,
    )

    for h in hooks:
        h.remove()

    # --- Compute weight-only, data-aware, and input-spectrum energy curves ---
    weight_curves = {}
    data_curves = {}
    input_curves = {}
    rank_at_90 = {}  # name -> (k_input, k_weight, k_data, max_rank)

    for name, W in weights.items():
        if cx_accum[name] is None or n_samples[name] == 0:
            continue
        Cx = cx_accum[name] / n_samples[name]
        try:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            print(f"  SVD failed for {name}: {e}")
            continue
        V = Vh.T
        # diag(Vᵀ Cx V) — projection energy of inputs onto each right singular vector
        proj = torch.diagonal(V.T @ Cx @ V).clamp(min=0)
        sigma2 = (S ** 2)
        weight_energy = sigma2.numpy()
        data_energy = (sigma2 * proj).numpy()

        # Input spectrum: eigenvalues of the input covariance Cx (descending).
        # These are σᵢ²(X)/N where σᵢ(X) are singular values of the input
        # matrix X — same cumulative-energy curve as SVD of X itself.
        try:
            input_eigs = torch.linalg.eigvalsh(Cx).flip(0).clamp(min=0).numpy()
        except Exception as e:
            print(f"  eigvalsh failed for {name}: {e}")
            input_eigs = None

        if weight_energy.sum() <= 0 or data_energy.sum() <= 0:
            continue
        w_cum = np.concatenate(([0.0], np.cumsum(weight_energy))) / weight_energy.sum() * 100
        d_cum = np.concatenate(([0.0], np.cumsum(data_energy))) / data_energy.sum() * 100
        weight_curves[name] = w_cum
        data_curves[name] = d_cum

        if input_eigs is not None and input_eigs.sum() > 0:
            i_cum = np.concatenate(([0.0], np.cumsum(input_eigs))) / input_eigs.sum() * 100
            input_curves[name] = i_cum
            ki = int(np.searchsorted(i_cum, 90.0))
        else:
            ki = -1

        # k at which we cross 90% energy for each curve
        kw = int(np.searchsorted(w_cum, 90.0))
        kd = int(np.searchsorted(d_cum, 90.0))
        rank_at_90[name] = (ki, kw, kd, len(S))

    if not data_curves:
        print("No data-aware curves produced (no activations collected).")
        return

    # --- Plot: weight-only vs data-aware, side-by-side ---
    short_names = _trim_common_prefix(list(data_curves.keys()))
    n = len(data_curves)
    show_labels = n <= 30
    cmap = plt.get_cmap('viridis')

    fig, (ax_w, ax_d) = plt.subplots(1, 2, figsize=(16, 6))
    for i, (name, short) in enumerate(zip(data_curves.keys(), short_names)):
        color = cmap(i / max(1, n - 1)) if n > 1 else cmap(0.5)
        ax_w.plot(range(len(weight_curves[name])), weight_curves[name],
                  color=color, alpha=0.6, linewidth=1.0,
                  label=short if show_labels else None)
        ax_d.plot(range(len(data_curves[name])), data_curves[name],
                  color=color, alpha=0.6, linewidth=1.0,
                  label=short if show_labels else None)

    # Independent y-axis scaling: each panel auto-fits to its own data so
    # subtle structure isn't squashed into a small range when the other panel
    # spans a different scale.
    w_max = max(c.max() for c in weight_curves.values())
    d_max = max(c.max() for c in data_curves.values())
    for ax, title, ymax in [
        (ax_w, 'Weight-only (Frobenius)', w_max),
        (ax_d, 'Data-aware (test activations)', d_max),
    ]:
        ax.axhline(90, color='red', linestyle='--', alpha=0.5, label='90% energy')
        ax.axhline(99, color='red', linestyle=':', alpha=0.5, label='99% energy')
        ax.set_xlabel('Rank K')
        ax.set_ylabel('% Cumulative Energy in Top-K')
        ax.set_ylim(0, max(105.0, ymax * 1.05))
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.legend(fontsize=6 if show_labels else 9, loc='lower right',
                  ncol=2 if show_labels else 1)
    fig.suptitle(f'Expert Spectral Energy: Weight vs Data-aware ({n} layers, '
                 f'{max(n_samples.values())} tokens)')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'expert_data_aware_energy.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Data-aware energy plot saved: {save_path}")

    # --- Plot: input | weight | data-aware (separate PNG for comparison) ---
    if input_curves:
        fig3, (ax_in, ax_w3, ax_d3) = plt.subplots(1, 3, figsize=(20, 6))
        for i, (name, short) in enumerate(zip(data_curves.keys(), short_names)):
            color = cmap(i / max(1, n - 1)) if n > 1 else cmap(0.5)
            if name in input_curves:
                ax_in.plot(range(len(input_curves[name])), input_curves[name],
                           color=color, alpha=0.6, linewidth=1.0,
                           label=short if show_labels else None)
            ax_w3.plot(range(len(weight_curves[name])), weight_curves[name],
                       color=color, alpha=0.6, linewidth=1.0,
                       label=short if show_labels else None)
            ax_d3.plot(range(len(data_curves[name])), data_curves[name],
                       color=color, alpha=0.6, linewidth=1.0,
                       label=short if show_labels else None)
        i_max = max(c.max() for c in input_curves.values())
        for ax, title, ymax in [
            (ax_in, 'Input spectrum (Cx eigenvalues)', i_max),
            (ax_w3, 'Weight-only (Frobenius)', w_max),
            (ax_d3, 'Data-aware (test activations)', d_max),
        ]:
            ax.axhline(90, color='red', linestyle='--', alpha=0.5, label='90% energy')
            ax.axhline(99, color='red', linestyle=':', alpha=0.5, label='99% energy')
            ax.set_xlabel('Rank K')
            ax.set_ylabel('% Cumulative Energy in Top-K')
            ax.set_ylim(0, max(105.0, ymax * 1.05))
            ax.grid(True, alpha=0.3)
            ax.set_title(title)
            ax.legend(fontsize=6 if show_labels else 9, loc='lower right',
                      ncol=2 if show_labels else 1)
        fig3.suptitle(f'Expert Spectra: Input vs Weight vs Data-aware ({n} layers, '
                       f'{max(n_samples.values())} tokens)')
        plt.tight_layout()
        cmp_save_path = os.path.join(output_dir, 'expert_input_spectrum_comparison.png')
        plt.savefig(cmp_save_path, dpi=150)
        plt.close()
        print(f"  Input spectrum comparison plot saved: {cmp_save_path}")

    # --- Summary table: rank to reach 90% energy, input vs weight vs data ---
    print(f"\n  Rank-at-90%-energy: input -> weight -> data-aware (max rank)")
    print(f"  {'layer':<70}  {'k_in':>5}  {'k_w':>5}  {'k_d':>5}  {'max':>5}")
    short_for_table = _trim_common_prefix(list(rank_at_90.keys()))
    for (name, (ki, kw, kd, mr)), short in zip(rank_at_90.items(), short_for_table):
        ki_str = f'{ki}' if ki >= 0 else '-'
        print(f"  {short:<70}  {ki_str:>5}  {kw:>5}  {kd:>5}  {mr:>5}")

    json_path = os.path.join(output_dir, 'expert_data_aware_energy.json')
    with open(json_path, 'w') as f:
        json.dump({
            'weight_only': {n: c.tolist() for n, c in weight_curves.items()},
            'data_aware': {n: c.tolist() for n, c in data_curves.items()},
            'input_spectrum': {n: c.tolist() for n, c in input_curves.items()},
            'rank_at_90_pct': {n: {'input': ki, 'weight': kw, 'data': kd,
                                    'max': mr}
                                for n, (ki, kw, kd, mr) in rank_at_90.items()},
            'tokens_per_layer': n_samples,
        }, f, indent=2)
    print(f"  Data-aware energy data saved: {json_path}")


def plot_routing_distribution(model, encoder, all_test, modalities_per_task,
                              args, device, output_dir, max_batches=10**9):
    """Per-MoE-layer routing-distribution heatmaps, broken down by (task, modality).

    Routers in this codebase are *modality-specific*: each SeqMoE layer holds a
    `routers` ModuleDict keyed by modality type (ts/txt/cxr/ecg/cc/mlo/...),
    routing into a shared expert pool. Each router emits sparse top-k gates
    over the expert pool per token.

    For every router call during a test forward pass, we accumulate two
    quantities per (layer, task, modality, expert):
      - activation_ratio: fraction of tokens where this expert was in top-k
      - mean_gate_weight: average sparse gate weight assigned

    Output: one figure per MoE layer with two heatmaps (rows = task|modality,
    cols = expert index) plus a raw-data JSON.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping routing distribution plot")
        return

    # --- Discover routers and parse (layer_prefix, modality) from each name ---
    router_info = {}  # full_name -> (layer_prefix, modality_label)
    for name, module in model.named_modules():
        if 'Router' not in type(module).__name__:
            continue
        parts = name.split('.')
        if 'routers' in parts:
            ri = parts.index('routers')
            layer_prefix = '.'.join(parts[:ri])
            modality = parts[ri + 1] if ri + 1 < len(parts) else 'default'
        elif 'default_router' in parts:
            ri = parts.index('default_router')
            layer_prefix = '.'.join(parts[:ri])
            modality = 'default'
        else:
            continue
        router_info[name] = (layer_prefix, modality)

    if not router_info:
        print("No ModalityRouter modules found.")
        return

    current = {'task': 'unknown'}
    accum = {}  # layer_prefix -> {(task, modality): {'gate_sum', 'active_count', 'tokens'}}

    hooks = []
    for name, module in model.named_modules():
        if name not in router_info:
            continue
        layer_prefix, modality = router_info[name]
        accum.setdefault(layer_prefix, {})

        def make_hook(lp, mod):
            def hook_fn(m, inp, out):
                with torch.no_grad():
                    gates = out[0] if isinstance(out, tuple) else out
                    if gates is None:
                        return
                    g = gates.float().detach().cpu()
                    key = (current['task'], mod)
                    b = accum[lp].get(key)
                    if b is None:
                        b = {
                            'gate_sum': torch.zeros(g.shape[-1]),
                            'active_count': torch.zeros(g.shape[-1]),
                            'tokens': 0,
                        }
                        accum[lp][key] = b
                    b['gate_sum'] += g.sum(dim=0)
                    b['active_count'] += (g > 0).float().sum(dim=0)
                    b['tokens'] += g.shape[0]
            return hook_fn
        hooks.append(module.register_forward_hook(make_hook(layer_prefix, modality)))

    # Drive forward passes; stamp the active task into hook state at each task boundary.
    def _set_task(t):
        current['task'] = t
    _run_test_forward_passes(
        model, encoder, all_test, modalities_per_task,
        args, device, max_batches=max_batches, on_task_start=_set_task,
    )

    for h in hooks:
        h.remove()

    # --- One figure per MoE layer ---
    json_payload = {}
    for layer_prefix in sorted(accum.keys()):
        layer_data = accum[layer_prefix]
        if not layer_data:
            continue
        sample = next(iter(layer_data.values()))
        n_experts = sample['gate_sum'].shape[0]

        # Aggregate across tasks per modality (routers are modality-specific:
        # the same modality's bars across tasks share router weights, so the
        # functional summary of a router is its per-modality routing pattern).
        # Per-task breakdown is preserved in the JSON output.
        mod_agg = {}
        for (tsk, mod), b in layer_data.items():
            agg = mod_agg.setdefault(mod, {
                'active_count': np.zeros(n_experts),
                'gate_sum': np.zeros(n_experts),
                'tokens': 0,
                'tasks': [],
            })
            agg['active_count'] += b['active_count'].numpy()
            agg['gate_sum'] += b['gate_sum'].numpy()
            agg['tokens'] += b['tokens']
            agg['tasks'].append(tsk)

        modalities = sorted(mod_agg.keys())
        n_mod = len(modalities)
        cmap = plt.get_cmap('tab10' if n_mod <= 10 else 'tab20')
        mod_colors = {m: cmap(i % cmap.N) for i, m in enumerate(modalities)}

        # Narrow bars: leave half of each unit free for inter-expert spacing.
        bar_width = 0.5 / max(1, n_mod)

        fig_width = max(12, n_experts * 0.9)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 5.5))

        x = np.arange(n_experts)
        for j, mod in enumerate(modalities):
            offset = (j - (n_mod - 1) / 2) * bar_width
            agg = mod_agg[mod]
            tokens = max(agg['tokens'], 1)
            ratios = agg['active_count'] / tokens * 100.0
            gate_w = agg['gate_sum'] / tokens
            ax1.bar(x + offset, ratios, bar_width,
                    color=mod_colors[mod], alpha=0.9, label=mod,
                    edgecolor='none')
            ax2.bar(x + offset, gate_w, bar_width,
                    color=mod_colors[mod], alpha=0.9, label=mod,
                    edgecolor='none')

        for ax, ylabel, title in [
            (ax1, '% tokens routed to expert', 'Activation ratio'),
            (ax2, 'mean gate weight', 'Mean gate weight (sparse)'),
        ]:
            ax.set_xlabel('Expert index')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.grid(True, axis='y', alpha=0.25)

        # Single shared legend below both panels.
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   ncol=min(n_mod, 5), fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f'Routing distribution — {layer_prefix}', fontsize=12)
        plt.tight_layout(rect=[0, 0.04, 1, 0.97])

        layer_short = layer_prefix.replace('.', '_')
        # Encode routing-decision counts in the filename: collapse to a single
        # suffix when all modalities saw the same n; otherwise list per-modality.
        counts = [mod_agg[m]['tokens'] for m in modalities]
        if counts and len(set(counts)) == 1:
            count_suffix = f'n{counts[0]}'
        else:
            count_suffix = '_'.join(f'{m}_n{mod_agg[m]["tokens"]}'
                                     for m in modalities)
        save_path = os.path.join(
            output_dir, f'routing_{layer_short}_{count_suffix}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Routing distribution saved: {save_path}")

        # --- Per-task plot: one row of (ratio | gate-weight) subplots per task,
        # bars colored by modality (same palette as the aggregated plot). Skip
        # when there's only one task — it would duplicate the aggregated plot.
        by_task = {}
        for (tsk, mod), b in layer_data.items():
            tk = max(b['tokens'], 1)
            by_task.setdefault(tsk, {})[mod] = {
                'ratios': (b['active_count'].numpy() / tk) * 100.0,
                'gate_w': b['gate_sum'].numpy() / tk,
                'tokens': int(b['tokens']),
            }

        if len(by_task) > 1:
            tasks = sorted(by_task.keys())
            n_tasks = len(tasks)

            fig_height_pt = max(3.0, 2.6 * n_tasks)
            fig_pt, axes_pt = plt.subplots(
                n_tasks, 2, figsize=(fig_width, fig_height_pt),
                sharex=True, squeeze=False,
            )

            legend_handles = {}
            for row, task in enumerate(tasks):
                task_mods = by_task[task]
                ax_r = axes_pt[row, 0]
                ax_g = axes_pt[row, 1]
                for j, mod in enumerate(modalities):
                    if mod not in task_mods:
                        continue
                    offset = (j - (n_mod - 1) / 2) * bar_width
                    d = task_mods[mod]
                    bars = ax_r.bar(x + offset, d['ratios'], bar_width,
                                    color=mod_colors[mod], alpha=0.9,
                                    label=mod, edgecolor='none')
                    ax_g.bar(x + offset, d['gate_w'], bar_width,
                             color=mod_colors[mod], alpha=0.9,
                             label=mod, edgecolor='none')
                    legend_handles.setdefault(mod, bars[0])

                ax_r.set_ylabel(f'{task}\n% routed', fontsize=10)
                ax_g.set_ylabel(f'{task}\nmean gate w', fontsize=10)
                ax_r.set_xticks(x)
                ax_g.set_xticks(x)
                ax_r.grid(True, axis='y', alpha=0.25)
                ax_g.grid(True, axis='y', alpha=0.25)
                if row == 0:
                    ax_r.set_title('Activation ratio')
                    ax_g.set_title('Mean gate weight (sparse)')
                if row == n_tasks - 1:
                    ax_r.set_xlabel('Expert index')
                    ax_g.set_xlabel('Expert index')

            fig_pt.legend(
                list(legend_handles.values()), list(legend_handles.keys()),
                loc='lower center', ncol=min(n_mod, 5), fontsize=9,
                bbox_to_anchor=(0.5, -0.02),
            )
            fig_pt.suptitle(f'Routing distribution by task — {layer_prefix}',
                             fontsize=12)
            plt.tight_layout(rect=[0, 0.04, 1, 0.97])

            # Per-task counts in the filename: each task lists either a single
            # `n<count>` (if all its modalities saw the same number) or a
            # `n<min>-<max>` range when they diverge.
            task_count_parts = []
            for task in tasks:
                tcounts = [d['tokens'] for d in by_task[task].values()]
                if len(set(tcounts)) == 1:
                    task_count_parts.append(f'{task}_n{tcounts[0]}')
                else:
                    task_count_parts.append(
                        f'{task}_n{min(tcounts)}-{max(tcounts)}')
            per_task_suffix = '_'.join(task_count_parts)

            per_task_save_path = os.path.join(
                output_dir,
                f'routing_{layer_short}_per_task_{per_task_suffix}.png',
            )
            plt.savefig(per_task_save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Per-task routing distribution saved: {per_task_save_path}")

        # JSON keeps per-(task, modality) granularity for downstream analysis.
        json_payload[layer_prefix] = {
            f'{tsk}|{mod}': {
                'activation_ratio_pct': ((b['active_count'].numpy()
                                          / max(b['tokens'], 1)) * 100.0).tolist(),
                'mean_gate_weight': (b['gate_sum'].numpy()
                                      / max(b['tokens'], 1)).tolist(),
                'tokens': int(b['tokens']),
            }
            for (tsk, mod), b in layer_data.items()
        }

    json_path = os.path.join(output_dir, 'routing_distribution.json')
    with open(json_path, 'w') as f:
        json.dump(json_payload, f, indent=2)
    print(f"  Routing data saved: {json_path}")


def compute_router_ranks(model):
    """Compute effective rank of router w_gate and w_noise matrices.

    Returns a list of dicts with router info.
    """
    sd = model.state_dict()
    router_info = []

    for key in sorted(sd):
        if '.moe.routers.' not in key and '.moe.gate.' not in key:
            continue
        W = sd[key]
        if W.dim() < 2:
            continue

        S = torch.linalg.svdvals(W.float())
        S = S[S > 1e-8]

        if len(S) == 0:
            router_info.append({
                'key': key,
                'shape': list(W.shape),
                'params': W.numel(),
                'eff_rank': 0.0,
                'max_rank': min(W.shape),
                'rank_90_energy': 0,
                'rank_99_energy': 0,
                'top_sv': 0.0,
                'sv_ratio_1_2': float('inf'),
                'note': 'all singular values < 1e-8',
            })
            continue

        # Effective rank (entropy-based)
        p = S / S.sum()
        eff_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-10))).item()

        # Fraction of energy in top-k
        total_energy = (S ** 2).sum().item()
        cum_energy = torch.cumsum(S ** 2, dim=0)
        ratio = cum_energy / total_energy
        idx_90 = (ratio >= 0.90).nonzero(as_tuple=True)[0]
        idx_99 = (ratio >= 0.99).nonzero(as_tuple=True)[0]
        rank_90 = int(idx_90[0].item()) + 1 if len(idx_90) > 0 else len(S)
        rank_99 = int(idx_99[0].item()) + 1 if len(idx_99) > 0 else len(S)

        router_info.append({
            'key': key,
            'shape': list(W.shape),
            'params': W.numel(),
            'eff_rank': eff_rank,
            'max_rank': min(W.shape),
            'rank_90_energy': rank_90,
            'rank_99_energy': rank_99,
            'top_sv': S[0].item(),
            'sv_ratio_1_2': (S[0] / S[1]).item() if len(S) > 1 else float('inf'),
        })

    return router_info


def evaluate(model, encoder, test, modalities, args, device):
    """Run evaluation and return metrics dict per task."""
    model.eval()
    for enc in encoder.values():
        enc.eval()

    task_names = {'MOR': 'mortality', 'RAD': 'readmission'}
    missing_embeddings = torch.nn.ParameterDict()
    results = {}

    with torch.no_grad():
        for ii in range(len(test)):
            task = modalities[int(ii)][0].split('_')[1]
            model.to_logits = model.to_logitslist[ii]
            eval_logits = []
            eval_labels = []

            for jj in tqdm(test[ii], desc=f'Eval {task}', leave=False):
                if task in ['IHM', 'PHENO', 'LOS']:
                    ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts, \
                        input_ids_sequences, attn_mask_sequences, text_emb, \
                        note_time, note_time_mask, cxr_feats, cxr_time, \
                        cxr_time_mask, ecg_feats, ecg_time, ecg_time_mask, \
                        label, cxr_missing, text_missing, ecg_missing = jj
                    embeddings = encoder[task](
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        input_ids_sequences=input_ids_sequences,
                        attn_mask_sequences=attn_mask_sequences,
                        text_emb=text_emb,
                        note_time_list=note_time,
                        note_time_mask_list=note_time_mask,
                        cxr_feats=cxr_feats,
                        cxr_time=cxr_time,
                        cxr_time_mask=cxr_time_mask,
                        ecg_feats=ecg_feats,
                        ecg_time=ecg_time,
                        ecg_time_mask=ecg_time_mask,
                        labels=label, reg_ts=reg_ts,
                        cxr_missing=cxr_missing,
                        text_missing=text_missing,
                        ecg_missing=ecg_missing,
                        modalities=modalities[int(ii)]
                    )
                elif task in ['MOR', 'RAD']:
                    codes, types, timestamps, ages, genders, ethnicities, label = \
                        jj['codes'], jj['types'], jj['timestamps'], jj['age'], \
                        jj['gender'], jj['ethnicity'], jj[task_names[task]].long()
                    embeddings = encoder[task](
                        codes=codes, types=types, timestamps=timestamps,
                        ages=ages, genders=genders, ethnicities=ethnicities,
                        modalities=modalities[int(ii)]
                    )
                elif task.lower() in ['birads', 'risk', 'density']:
                    idx, label, embed_2dcc, embed_2dmlo, embed_cc, embed_mlo, all_views = jj
                    embeddings = encoder[task](
                        embed_cc=embed_cc, embed_mlo=embed_mlo,
                        embed_2dcc=embed_2dcc, embed_2dmlo=embed_2dmlo,
                        all_views=all_views,
                        modalities=modalities[int(ii)], task=task
                    )

                indict = {}
                for i in range(len(modalities[ii])):
                    indict[modalities[ii][i]] = embeddings[modalities[ii][i]].float().to(device)

                out, balance_loss = model(indict, task=task)

                if 'PHENO' in modalities[int(ii)][0]:
                    logit = torch.nn.functional.sigmoid(out)
                elif 'birads' in modalities[int(ii)][0].lower() or 'density' in modalities[int(ii)][0].lower():
                    logit = torch.nn.functional.softmax(out, dim=-1)
                else:
                    logit = torch.nn.functional.softmax(out, dim=-1)[:, 1]

                eval_logits += logit.cpu().numpy().tolist()
                eval_labels += label.cpu().numpy().tolist()

            all_logits = np.array(eval_logits)
            all_label = np.array(eval_labels)

            eval_vals = {}
            if 'PHENO' in modalities[int(ii)][0]:
                all_pred = np.where(all_logits > 0.5, 1, 0)
                eval_vals = metrics_multilabel(all_label, all_logits, verbose=0)
                eval_vals['macro_f1'] = f1_score(all_label, all_pred, average='macro')
                eval_vals['primary_metric'] = float(eval_vals['auc_scores'].mean())
                eval_vals['metric_name'] = 'auc_mean'
            elif 'birads' in modalities[int(ii)][0].lower() or 'density' in modalities[int(ii)][0].lower():
                eval_vals = metrics_multiclass(all_label, all_logits, verbose=0)
                all_pred = np.argmax(all_logits, axis=1)
                eval_vals['macro_f1'] = f1_score(all_label, all_pred, average='macro')
                eval_vals['accuracy'] = accuracy_score(all_label, all_pred)
                eval_vals['primary_metric'] = float(eval_vals['ave_auc_macro'])
                eval_vals['metric_name'] = 'ave_auc_macro'
            else:
                all_pred = np.where(all_logits > 0.5, 1, 0)
                eval_vals['auc'] = roc_auc_score(all_label, all_logits)
                precisions, recalls, _ = precision_recall_curve(all_label, all_logits)
                eval_vals['auprc'] = auc(recalls, precisions)
                eval_vals['f1'] = f1_score(all_label, all_pred)
                eval_vals['accuracy'] = accuracy_score(all_label, all_pred)
                eval_vals['primary_metric'] = eval_vals['auc']
                eval_vals['metric_name'] = 'auc'

            # Convert numpy types to float for JSON serialization
            results[task] = {k: float(v) if isinstance(v, (np.floating, float)) else v
                             for k, v in eval_vals.items()
                             if isinstance(v, (int, float, np.integer, np.floating, str))}

    return results


def main():
    # --- Parse args (reuse mimiciv_tasks args + add our own) ---
    # We inject our extra args before parse_args sees them
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument('--model_path', type=str, required=True,
                              help='Path to model checkpoint (.pt)')
    extra_parser.add_argument('--encoder_path', type=str, nargs='+', required=True,
                              help='Path(s) to encoder checkpoint(s). For multi-task with different encoders, '
                                   'provide one per task in the same order as tasks in --task (e.g., --task ihm-birads '
                                   '--encoder_path ihm_encoder.pt birads_encoder.pt)')
    extra_parser.add_argument('--ranks', nargs='+', default=['1', '2', '4', '8', '16', '32', '64', 'full'],
                              help='Ranks to evaluate. Use "full" for no truncation.')
    extra_parser.add_argument('--output_dir', type=str, default='analysis/analysis_results/lowrank_eval',
                              help='Directory to save results')
    extra_args, remaining = extra_parser.parse_known_args()

    # Put remaining args back for parse_args
    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()
    args.num_train_epochs = 0  # no training

    set_seed(args.seed)
    if args.fp16:
        args.mixed_precision = "fp16"
    else:
        args.mixed_precision = "no"
    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=args.cpu)
    device = accelerator.device

    # --- Build modalities (same logic as mimiciv_tasks.py main) ---
    task_mods_dict = {
        'ihm_mod': args.ihm_mod, 'los_mod': args.los_mod, 'pheno_mod': args.pheno_mod,
        'ihm-los-pheno_mod': args.ihm_mod+'_'+args.los_mod+'_'+args.pheno_mod,
        'ihm-los_mod': args.ihm_mod+'_'+args.los_mod,
        'ihm-pheno_mod': args.ihm_mod+'_'+args.pheno_mod,
        'los-pheno_mod': args.los_mod+'_'+args.pheno_mod,
        'readmission_mod': args.rad_mod, 'mortality_mod': args.mor_mod,
        'mortality-readmission_mod': args.mor_mod+'_'+args.rad_mod,
        'ihm-mortality_mod': args.ihm_mod+'_'+args.mor_mod,
        'los-readmission_mod': args.los_mod+'_'+args.rad_mod,
        'ihm-readmission_mod': args.ihm_mod+'_'+args.rad_mod,
        'los-mortality_mod': args.los_mod+'_'+args.mor_mod,
        'ihm-los-mortality_mod': args.ihm_mod+'_'+args.los_mod+'_'+args.mor_mod,
        'ihm-los-mortality-readmission_mod': args.ihm_mod+'_'+args.los_mod+'_'+args.mor_mod+'_'+args.rad_mod,
        'birads_mod': args.birads_mod, 'risk_mod': args.risk_mod, 'density_mod': args.density_mod,
        'birads-risk-density_mod': args.birads_mod+'_'+args.risk_mod+'_'+args.density_mod,
        'ihm-birads_mod': args.ihm_mod+'_'+args.birads_mod,
        'birads-risk_mod': args.birads_mod+'_'+args.risk_mod,
        'birads-density_mod': args.birads_mod+'_'+args.density_mod,
        'risk-density_mod': args.risk_mod+'_'+args.density_mod,
        'ihm-los-pheno-birads-risk-density_mod': args.ihm_mod+'_'+args.los_mod+'_'+args.pheno_mod+'_'+args.birads_mod+'_'+args.risk_mod+'_'+args.density_mod,
        'ihm-los-pheno-mortality-readmission-birads-risk-density_mod': args.ihm_mod+'_'+args.los_mod+'_'+args.pheno_mod+'_'+args.mor_mod+'_'+args.rad_mod+'_'+args.birads_mod+'_'+args.risk_mod+'_'+args.density_mod
    }
    task_mod_key = f'{args.task}_mod'

    modalities = set()
    modeltype = {}
    if len(args.ihm_mod) != 0 and 'ihm' in args.task.split("-"):
        modeltype['ihm'] = '_'.join(sorted(args.ihm_mod.split("-")))
        for e in args.ihm_mod.split("-"):
            modalities.add(e)
    if len(args.los_mod) != 0 and 'los' in args.task.split("-"):
        modeltype['los'] = '_'.join(sorted(args.los_mod.split("-")))
        for e in args.los_mod.split("-"):
            modalities.add(e)
    if len(args.pheno_mod) != 0 and 'pheno' in args.task.split("-"):
        modeltype['pheno'] = '_'.join(sorted(args.pheno_mod.split("-")))
        for e in args.pheno_mod.split("-"):
            modalities.add(e)
    if len(args.rad_mod) != 0 and 'readmission' in args.task.split("-"):
        modeltype['readmission'] = '_'.join(sorted(args.rad_mod.split("-")))
        for e in args.rad_mod.split("-"):
            modalities.add(e)
    if len(args.mor_mod) != 0 and 'mortality' in args.task.split("-"):
        modeltype['mortality'] = '_'.join(sorted(args.mor_mod.split("-")))
        for e in args.mor_mod.split("-"):
            modalities.add(e)
    if len(args.birads_mod) != 0 and 'birads' in args.task.split("-"):
        modeltype['birads'] = '_'.join(sorted(args.birads_mod.split("-")))
        for e in args.birads_mod.split("-"):
            modalities.add(e)
    if len(args.risk_mod) != 0 and 'risk' in args.task.split("-"):
        modeltype['risk'] = '_'.join(sorted(args.risk_mod.split("-")))
        for e in args.risk_mod.split("-"):
            modalities.add(e)
    if len(args.density_mod) != 0 and 'density' in args.task.split("-"):
        modeltype['density'] = '_'.join(sorted(args.density_mod.split("-")))
        for e in args.density_mod.split("-"):
            modalities.add(e)

    if 'Text' in modalities:
        BioBert, BioBertConfig, tokenizer = loadBert(args, device)
    else:
        tokenizer = None
        BioBert = None

    (
        all_train, all_valid, all_test, criterion,
        modalities_per_task, train_weights, all_encoders, logits, all_modalities,
    ) = setup_tasks_and_modalities(
        args=args, device=device, tokenizer=tokenizer,
        modeltype=modeltype, modalities=modalities, BioBert=BioBert,
    )

    # --- Load model and encoder from checkpoints ---
    print(f"\nLoading model: {extra_args.model_path}")
    model = torch.load(extra_args.model_path, map_location=device)
    model.to_logitslist = model.to_logitslist.to(device)

    # --- Print learned residual gate (alpha) if present ---
    for name, param in model.named_parameters():
        if 'residual_gate' in name:
            alpha = torch.sigmoid(param).item()
            print(f"  {name}: raw={param.item():.4f}, alpha=sigmoid(raw)={alpha:.4f}")
            print(f"    -> residual weight: {alpha:.2%}, MoE weight: {1-alpha:.2%}")

    # Load encoder(s) — one per task or single shared encoder
    task_keys = list(all_encoders.keys())  # e.g. ['IHM', 'BIRADS']
    encoder_paths = extra_args.encoder_path

    if len(encoder_paths) == 1:
        # Single encoder shared across all tasks
        print(f"Loading shared encoder: {encoder_paths[0]}")
        encoder_loaded = torch.load(encoder_paths[0], map_location=device)
        encoder_loaded.to(device)
        for task_key in task_keys:
            all_encoders[task_key] = encoder_loaded
    elif len(encoder_paths) == len(task_keys):
        # One encoder per task, in order of tasks from --task arg
        for task_key, enc_path in zip(task_keys, encoder_paths):
            print(f"Loading encoder for {task_key}: {enc_path}")
            enc = torch.load(enc_path, map_location=device)
            enc.to(device)
            all_encoders[task_key] = enc
    else:
        raise ValueError(
            f"Got {len(encoder_paths)} encoder path(s) but {len(task_keys)} tasks ({task_keys}). "
            f"Provide either 1 (shared) or {len(task_keys)} encoder paths."
        )

    # --- Mirror checkpoint dir structure under output_dir ---
    run_subdir = derive_run_subdir(extra_args.model_path)
    extra_args.output_dir = os.path.join(extra_args.output_dir, run_subdir)
    print(f"Analysis output directory: {extra_args.output_dir}")

    # --- Evaluate at each rank ---
    os.makedirs(extra_args.output_dir, exist_ok=True)

    # --- Singular-value spectrum across all 2D layers (pre-truncation) ---
    print(f"\n{'='*60}")
    print("  Singular-Value Spectrum")
    print(f"{'='*60}")
    plot_singular_value_spectrum(model, extra_args.output_dir)

    # --- Cumulative spectral energy vs rank for expert layers ---
    print(f"\n{'='*60}")
    print("  Expert Spectral Energy vs Rank")
    print(f"{'='*60}")
    plot_expert_energy_curves(model, extra_args.output_dir)

    # --- Data-aware energy: which weight ranks the test data actually excites ---
    print(f"\n{'='*60}")
    print("  Data-Aware Spectral Energy (test activations)")
    print(f"{'='*60}")
    plot_data_aware_energy_curves(
        model, all_encoders, all_test, modalities_per_task,
        args, device, extra_args.output_dir, max_batches=10,
    )

    # --- Routing distribution per (task, modality) ---
    print(f"\n{'='*60}")
    print("  Routing Distribution (per task × modality)")
    print(f"{'='*60}")
    plot_routing_distribution(
        model, all_encoders, all_test, modalities_per_task,
        args, device, extra_args.output_dir, max_batches=10,
    )

    # Parse ranks
    rank_values = []
    for r in extra_args.ranks:
        if r.lower() == 'full':
            rank_values.append(None)  # None = no truncation
        else:
            rank_values.append(int(r))

    all_results = {}

    # --- Diagnose expert contribution (before any truncation) ---
    diagnose_expert_contribution(model, all_encoders, all_test,
                                 modalities_per_task, args, device, max_batches=10)

    # --- Router rank analysis (independent of SVD truncation) ---
    print(f"\n{'='*60}")
    print("  Router Rank Analysis")
    print(f"{'='*60}")
    router_info = compute_router_ranks(model)
    if router_info:
        for info in router_info:
            print(f"  {info['key']}")
            print(f"    shape={info['shape']}, params={info['params']}")
            print(f"    effective_rank={info['eff_rank']:.2f} / {info['max_rank']}")
            print(f"    rank@90%energy={info['rank_90_energy']}, rank@99%energy={info['rank_99_energy']}")
            print(f"    top_sv={info['top_sv']:.4f}, sv_ratio(1/2)={info['sv_ratio_1_2']:.2f}")
        # Save router analysis
        router_path = os.path.join(extra_args.output_dir, 'router_rank_analysis.json')
        with open(router_path, 'w') as f:
            json.dump(router_info, f, indent=2)
        print(f"  Router analysis saved to {router_path}")
    else:
        print("  No router weight matrices found.")

    for rank in rank_values:
        rank_label = 'full' if rank is None else str(rank)
        print(f"\n{'='*60}")
        print(f"  Evaluating rank = {rank_label}")
        print(f"{'='*60}")

        # Deep copy the model so we don't accumulate truncations
        model_copy = copy.deepcopy(model)

        if rank is not None:
            n_modified = apply_lowrank_to_experts(model_copy, rank)
            print(f"  Truncated {n_modified} expert weight matrices to rank {rank}")
        else:
            print(f"  Using full-rank weights (baseline)")

        # Parameter count at this rank
        full_params, lowrank_params, param_details = count_expert_params(model_copy, rank)
        compression = (1 - lowrank_params / full_params) * 100 if full_params > 0 else 0
        print(f"  Expert params: full={full_params:,}, at_rank={lowrank_params:,} ({compression:.1f}% compression)")
        if param_details:
            for key, shape, full_n, factored_n in param_details:
                short_key = key.split('.moe.')[-1]
                print(f"    {short_key}: {shape} {full_n:,} -> {factored_n:,}")

        results = evaluate(model_copy, all_encoders, all_test,
                           modalities_per_task, args, device)

        all_results[rank_label] = results
        all_results[rank_label]['_param_info'] = {
            'full_expert_params': full_params,
            'lowrank_expert_params': lowrank_params,
            'compression_pct': round(compression, 2),
        }

        # Print results
        for task, metrics in results.items():
            if task.startswith('_'):
                continue
            metric_name = metrics.get('metric_name', '?')
            primary = metrics.get('primary_metric', '?')
            print(f"  {task}: {metric_name}={primary:.4f}", end='')
            if 'f1' in metrics:
                print(f", f1={metrics['f1']:.4f}", end='')
            if 'auprc' in metrics:
                print(f", auprc={metrics['auprc']:.4f}", end='')
            print()

        del model_copy
        torch.cuda.empty_cache()

    # --- Save results ---
    results_path = os.path.join(extra_args.output_dir, 'lowrank_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Print summary table ---
    print(f"\n{'='*70}")
    print("SUMMARY: Primary metric by rank")
    print(f"{'='*70}")
    tasks = [k for k in next(iter(all_results.values())).keys() if not k.startswith('_')]
    header = f"{'Rank':>8}"
    for task in tasks:
        metric_name = all_results[list(all_results.keys())[0]][task].get('metric_name', 'metric')
        header += f"  {task} ({metric_name}):>18"
    # Simpler header
    print(f"{'Rank':>8}", end='')
    for task in tasks:
        print(f"  {task:>18}", end='')
    print()
    print("-" * (8 + 20 * len(tasks)))

    full_metrics = {}
    for task in tasks:
        if 'full' in all_results:
            full_metrics[task] = all_results['full'][task].get('primary_metric', 0)

    for rank_label in all_results:
        print(f"{rank_label:>8}", end='')
        for task in tasks:
            val = all_results[rank_label][task].get('primary_metric', 0)
            full_val = full_metrics.get(task, val)
            pct = (val / full_val * 100) if full_val != 0 else 0
            print(f"  {val:.4f} ({pct:5.1f}%)", end='')
        print()

    # --- Plot rank vs metric ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        for task in tasks:
            fig, ax = plt.subplots(figsize=(8, 5))
            ranks_numeric = []
            metrics_vals = []
            for rank_label, res in all_results.items():
                if rank_label == 'full':
                    continue
                ranks_numeric.append(int(rank_label))
                metrics_vals.append(res[task].get('primary_metric', 0))

            # Add full rank
            if 'full' in all_results:
                full_val = all_results['full'][task].get('primary_metric', 0)
                ax.axhline(y=full_val, color='gray', linestyle='--', alpha=0.7,
                           label=f'Full rank ({full_val:.4f})')

            if ranks_numeric:
                ax.plot(ranks_numeric, metrics_vals, 'bo-', linewidth=2, markersize=8)
                for r, v in zip(ranks_numeric, metrics_vals):
                    ax.annotate(f'{v:.3f}', (r, v), textcoords='offset points',
                                xytext=(0, 10), ha='center', fontsize=8)

            metric_name = all_results[list(all_results.keys())[0]][task].get('metric_name', 'metric')
            ax.set_xlabel('SVD Rank')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{task}: {metric_name} vs Expert Low-Rank Truncation')
            ax.set_xscale('log', base=2)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_path = os.path.join(extra_args.output_dir, f'lowrank_{task}.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Plot saved: {save_path}")
    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == '__main__':
    main()
