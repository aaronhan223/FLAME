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
            if outs:
                # Outputs vary in size across batches; truncate to common min length
                min_size = min(len(o) for o in outs if len(o) > 0)
                if min_size > 0:
                    expert_means[eidx] = torch.stack([o[:min_size] for o in outs if len(o) > 0]).mean(dim=0)

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

    # --- Evaluate at each rank ---
    os.makedirs(extra_args.output_dir, exist_ok=True)

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
