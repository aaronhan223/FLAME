"""
Evaluate FlexMoE (src.flexmoe.FlexMoE) models with low-rank approximations
of expert weights.

Parallel to src.analysis.eval_lowrank_experts (which targets fusemoe /
MULTCrossModel) but adapted to fastmoe's stacked-expert layout:

  * Each MoE layer's experts are stored as 3D tensors
    (num_expert, out_features, in_features) via `FMoELinear`. We apply SVD
    truncation to each expert slice independently, then restack.
  * Router weights (w_gate, w_noise) live under `.mlp.all_gates.*`.
  * Per-layer residual-vs-MoE diagnostics are skipped: Flex-MoE's
    TransformerEncoderLayer does not expose residual and MoE output as
    separable tensors the way TransformerCrossEncoderLayer does.

Usage (see run_lowrank_eval_flexmoe.sh):
    python -m src.analysis.eval_lowrank_experts_flexmoe \
        --model_path .../flexmoe/multitask/ihm/..._mod_drop_rate_0.0.pt \
        --encoder_path .../flexmoe/multitask/ihm/..._encoder.pt \
        --task ihm --ihm_mod TS-Text ... \
        --ranks 0 1 2 4 8 full \
        --output_dir .../analysis_results/lowrank_eval/flexmoe-ihm \
        --fusion_model flexmoe <remaining training-config args>
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
from src.flexmoe import FlexMoE  # noqa: F401 — needed for torch.load to resolve class
from transformers import set_seed
from accelerate import Accelerator

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# Key predicates — FlexMoE / fastmoe state_dict layout
# --------------------------------------------------------------------------
# Expert weights are stored in `FMoETransformerMLP.experts` (an `_Expert`
# module) with two `FMoELinear` tensors:
#   <prefix>.mlp.experts.htoh4.weight   shape (E, d_hidden, d_model)
#   <prefix>.mlp.experts.htoh4.bias     shape (E, d_hidden)
#   <prefix>.mlp.experts.h4toh.weight   shape (E, d_model, d_hidden)
#   <prefix>.mlp.experts.h4toh.bias     shape (E, d_model)
#
# Routers live under `all_gates`:
#   <prefix>.mlp.all_gates.<i>.w_gate   shape (d_model, num_expert)
#   <prefix>.mlp.all_gates.<i>.w_noise  shape (d_model, num_expert)
# and the active gate alias <prefix>.mlp.gate.w_gate / w_noise.

def _is_expert_weight(key):
    return '.mlp.experts.' in key and key.endswith('.weight')

def _is_expert_param(key):
    return '.mlp.experts.' in key

def _is_router_weight(key):
    # Match both `all_gates.<i>.` and the `gate.` alias but don't double-count:
    # we'll dedup by normalizing `mlp.gate.` (alias of all_gates.0) upstream.
    return ('.mlp.all_gates.' in key or '.mlp.gate.' in key) and (
        key.endswith('.w_gate') or key.endswith('.w_noise')
    )


# --------------------------------------------------------------------------
# Low-rank truncation
# --------------------------------------------------------------------------

def truncate_to_rank(W, rank):
    """SVD-truncate a 2D matrix to given rank. rank=0 returns zeros."""
    if rank == 0:
        return torch.zeros_like(W)
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    k = min(rank, len(S))
    W_approx = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]
    return W_approx.to(W.dtype)


def apply_lowrank_to_experts(model, rank):
    """Apply per-expert rank-k SVD to all stacked expert weights in-place.

    For each 3D expert weight [E, out, in] we SVD-truncate W[e] for every
    expert index e independently. Biases are untouched.
    Returns: number of expert matrices modified (counts per-expert slices).
    """
    sd = model.state_dict()
    modified = 0
    for key in list(sd.keys()):
        if not _is_expert_weight(key):
            continue
        W = sd[key]
        if W.dim() != 3:
            # Unexpected shape (shouldn't happen with fastmoe _Expert); skip.
            continue
        E = W.shape[0]
        W_new = W.clone()
        for e in range(E):
            W_new[e] = truncate_to_rank(W_new[e], rank)
        sd[key] = W_new
        modified += E
    model.load_state_dict(sd)
    return modified


def count_expert_params(model, rank=None):
    """Count expert params full vs. factored (per-expert SVD storage)."""
    sd = model.state_dict()
    full_params = 0
    lowrank_params = 0
    details = []

    for key in sorted(sd):
        if not _is_expert_param(key):
            continue
        W = sd[key]
        numel = W.numel()
        full_params += numel

        if rank is None or not key.endswith('.weight') or W.dim() != 3:
            # Biases and non-weight tensors: always count at full size.
            lowrank_params += numel
            continue

        E, out_f, in_f = W.shape
        k = min(rank, min(out_f, in_f))
        # Per-expert factored storage: U[out,k] + S[k] + Vh[k,in] = k*(out+in+1)
        factored = E * (k * (out_f + in_f + 1))
        lowrank_params += factored
        details.append((key, list(W.shape), numel, factored))
    return full_params, lowrank_params, details


# --------------------------------------------------------------------------
# Router rank analysis
# --------------------------------------------------------------------------

def compute_router_ranks(model):
    """Effective rank and energy-based ranks for every router matrix.

    Deduplicates the `mlp.gate.` alias (same storage as `mlp.all_gates.0.`).
    """
    sd = model.state_dict()
    seen_ids = set()
    router_info = []

    for key in sorted(sd):
        if not _is_router_weight(key):
            continue
        W = sd[key]
        # Skip the active-gate alias — same tensor as all_gates.0.
        if id(W) in seen_ids:
            continue
        seen_ids.add(id(W))
        if W.dim() < 2:
            continue

        S = torch.linalg.svdvals(W.float())
        S = S[S > 1e-8]
        if len(S) == 0:
            router_info.append({
                'key': key, 'shape': list(W.shape), 'params': W.numel(),
                'eff_rank': 0.0, 'max_rank': min(W.shape),
                'rank_90_energy': 0, 'rank_99_energy': 0,
                'top_sv': 0.0, 'sv_ratio_1_2': float('inf'),
                'note': 'all singular values < 1e-8',
            })
            continue

        p = S / S.sum()
        eff_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-10))).item()
        total_energy = (S ** 2).sum().item()
        cum_energy = torch.cumsum(S ** 2, dim=0)
        ratio = cum_energy / total_energy
        idx_90 = (ratio >= 0.90).nonzero(as_tuple=True)[0]
        idx_99 = (ratio >= 0.99).nonzero(as_tuple=True)[0]
        rank_90 = int(idx_90[0].item()) + 1 if len(idx_90) > 0 else len(S)
        rank_99 = int(idx_99[0].item()) + 1 if len(idx_99) > 0 else len(S)

        router_info.append({
            'key': key, 'shape': list(W.shape), 'params': W.numel(),
            'eff_rank': eff_rank, 'max_rank': min(W.shape),
            'rank_90_energy': rank_90, 'rank_99_energy': rank_99,
            'top_sv': S[0].item(),
            'sv_ratio_1_2': (S[0] / S[1]).item() if len(S) > 1 else float('inf'),
        })
    return router_info


# --------------------------------------------------------------------------
# Expert / MoE contribution diagnostics
# --------------------------------------------------------------------------

def diagnose_expert_contribution(model, encoder, test, modalities, args, device, max_batches=10):
    """Capture MoE and expert output statistics via forward hooks.

    Hooks fastmoe classes (`_Expert`, `FMoETransformerMLP`) and Flex-MoE
    layers (`TransformerEncoderLayer`). Per-expert disaggregation is not
    possible here because fastmoe processes all experts inside a single
    fused `FMoELinear` call; we report combined stacked-output stats plus
    per-layer output effective rank.
    """
    model.eval()
    for enc in encoder.values():
        enc.eval()

    expert_stats = {}       # _Expert module -> list of per-batch stat dicts
    moe_io_stats = []       # list of FMoETransformerMLP per-batch stats
    layer_stats = {}        # TransformerEncoderLayer -> list of per-batch stats

    expert_hooks, moe_hooks, layer_hooks = [], [], []

    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name == '_Expert':
            sl = expert_stats.setdefault(name, [])
            def make_hook(sl):
                def hook_fn(mod, inp, out):
                    with torch.no_grad():
                        i = inp[0].float() if isinstance(inp, tuple) else inp.float()
                        o = out.float()
                        sl.append({
                            'input_norm': i.norm().item(),
                            'output_norm': o.norm().item(),
                            'output_mean': o.mean().item(),
                            'output_std': o.std().item(),
                            'output_abs_mean': o.abs().mean().item(),
                            'input_abs_mean': i.abs().mean().item(),
                        })
                return hook_fn
            expert_hooks.append(module.register_forward_hook(make_hook(sl)))

        elif cls_name == 'FMoETransformerMLP':
            def make_moe_hook(name_=name):
                def hook_fn(mod, inp, out):
                    with torch.no_grad():
                        o = out.float() if isinstance(out, torch.Tensor) else out[0].float()
                        moe_io_stats.append({
                            'layer': name_,
                            'moe_output_norm': o.norm().item(),
                            'moe_output_abs_mean': o.abs().mean().item(),
                            'moe_output_std': o.std().item(),
                        })
                return hook_fn
            moe_hooks.append(module.register_forward_hook(make_moe_hook()))

        elif cls_name == 'TransformerEncoderLayer':
            sl = layer_stats.setdefault(name, [])
            def make_layer_hook(sl):
                def hook_fn(mod, inp, out):
                    with torch.no_grad():
                        # `out` is a list of per-modality tensors (B, T, D).
                        if not isinstance(out, (list, tuple)):
                            return
                        for t in out:
                            m = t.float()
                            m_flat = m.reshape(-1, m.shape[-1])
                            eff_rank = 0.0
                            if m_flat.shape[0] > 1:
                                S = torch.linalg.svdvals(m_flat)
                                S = S[S > 1e-8]
                                if len(S) > 0:
                                    p = S / S.sum()
                                    eff_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-10))).item()
                            sl.append({
                                'out_norm': m_flat.norm(dim=-1).mean().item(),
                                'eff_rank': eff_rank,
                                'max_rank': min(m_flat.shape),
                            })
                return hook_fn
            layer_hooks.append(module.register_forward_hook(make_layer_hook(sl)))

    # --- run a few batches (same data path as evaluate()) ---
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
                            cxr_feats=data['cxr_feats'], cxr_time=data['cxr_time'], cxr_time_mask=data['cxr_time_mask'],
                            ecg_feats=data['ecg_feats'], ecg_time=data['ecg_time'], ecg_time_mask=data['ecg_time_mask'],
                            labels=data['label'], reg_ts=data['reg_ts'],
                            cxr_missing=data['cxr_missing'], text_missing=data['text_missing'],
                            ecg_missing=data['ecg_missing'], modalities=modalities[ii],
                        )
                    elif task in ['BIRADS', 'RISK', 'DENSITY']:
                        encoded = encoder[task](
                            embed_cc=data['embed_cc'], embed_mlo=data['embed_mlo'],
                            embed_2dcc=data['embed_2dcc'], embed_2dmlo=data['embed_2dmlo'],
                            all_views=data['all_views'],
                            modalities=modalities[ii], task=task,
                        )
                    indict = {m: encoded[m].float().to(device) for m in modalities[ii]}
                    model(indict, task=task)
                except Exception as e:
                    print(f"  [diagnose] skip batch: {e}")
                batch_count += 1

    for h in expert_hooks + moe_hooks + layer_hooks:
        h.remove()

    # --- report ---
    print(f"\n{'='*60}\n  Expert Contribution Diagnosis (FlexMoE)\n{'='*60}")
    for name, stats in expert_stats.items():
        if not stats: continue
        avg_in = np.mean([s['input_norm']  for s in stats])
        avg_on = np.mean([s['output_norm'] for s in stats])
        avg_oa = np.mean([s['output_abs_mean'] for s in stats])
        avg_os = np.mean([s['output_std'] for s in stats])
        print(f"  {name}: |in|={avg_in:.3f}, |out|={avg_on:.3f}, |out|_mean={avg_oa:.5f}, |out|_std={avg_os:.5f}")

    if moe_io_stats:
        by_layer = {}
        for s in moe_io_stats:
            by_layer.setdefault(s['layer'], []).append(s)
        print(f"\n  MoE layer outputs:")
        for layer, lst in by_layer.items():
            print(f"    {layer}: norm={np.mean([s['moe_output_norm'] for s in lst]):.3f}, "
                  f"abs_mean={np.mean([s['moe_output_abs_mean'] for s in lst]):.5f}, "
                  f"std={np.mean([s['moe_output_std'] for s in lst]):.5f}")

    if layer_stats:
        print(f"\n  TransformerEncoderLayer outputs:")
        for name, lst in layer_stats.items():
            if not lst: continue
            print(f"    {name}: out_norm={np.mean([s['out_norm'] for s in lst]):.3f}, "
                  f"eff_rank={np.mean([s['eff_rank'] for s in lst]):.2f} / "
                  f"{int(np.mean([s['max_rank'] for s in lst]))}")

    return expert_stats, moe_io_stats, layer_stats


# --------------------------------------------------------------------------
# Evaluation (identical interface to the fusemoe version)
# --------------------------------------------------------------------------

def evaluate(model, encoder, test, modalities, args, device):
    model.eval()
    for enc in encoder.values():
        enc.eval()

    task_names = {'MOR': 'mortality', 'RAD': 'readmission'}
    results = {}

    with torch.no_grad():
        for ii in range(len(test)):
            task = modalities[int(ii)][0].split('_')[1]
            model.to_logits = model.to_logitslist[ii]
            eval_logits, eval_labels = [], []

            for jj in tqdm(test[ii], desc=f'Eval {task}', leave=False):
                if task in ['IHM', 'PHENO', 'LOS']:
                    ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts, \
                        input_ids_sequences, attn_mask_sequences, text_emb, \
                        note_time, note_time_mask, cxr_feats, cxr_time, \
                        cxr_time_mask, ecg_feats, ecg_time, ecg_time_mask, \
                        label, cxr_missing, text_missing, ecg_missing = jj
                    embeddings = encoder[task](
                        x_ts=ts_input_sequences, x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        input_ids_sequences=input_ids_sequences,
                        attn_mask_sequences=attn_mask_sequences,
                        text_emb=text_emb,
                        note_time_list=note_time, note_time_mask_list=note_time_mask,
                        cxr_feats=cxr_feats, cxr_time=cxr_time, cxr_time_mask=cxr_time_mask,
                        ecg_feats=ecg_feats, ecg_time=ecg_time, ecg_time_mask=ecg_time_mask,
                        labels=label, reg_ts=reg_ts,
                        cxr_missing=cxr_missing, text_missing=text_missing,
                        ecg_missing=ecg_missing, modalities=modalities[int(ii)],
                    )
                elif task in ['MOR', 'RAD']:
                    codes, types, timestamps, ages, genders, ethnicities, label = \
                        jj['codes'], jj['types'], jj['timestamps'], jj['age'], \
                        jj['gender'], jj['ethnicity'], jj[task_names[task]].long()
                    embeddings = encoder[task](
                        codes=codes, types=types, timestamps=timestamps,
                        ages=ages, genders=genders, ethnicities=ethnicities,
                        modalities=modalities[int(ii)],
                    )
                elif task.lower() in ['birads', 'risk', 'density']:
                    idx, label, embed_2dcc, embed_2dmlo, embed_cc, embed_mlo, all_views = jj
                    embeddings = encoder[task](
                        embed_cc=embed_cc, embed_mlo=embed_mlo,
                        embed_2dcc=embed_2dcc, embed_2dmlo=embed_2dmlo,
                        all_views=all_views,
                        modalities=modalities[int(ii)], task=task,
                    )

                indict = {m: embeddings[m].float().to(device) for m in modalities[ii]}
                out, _gate_loss = model(indict, task=task)

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

            results[task] = {k: float(v) if isinstance(v, (np.floating, float)) else v
                             for k, v in eval_vals.items()
                             if isinstance(v, (int, float, np.integer, np.floating, str))}
    return results


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument('--model_path', type=str, required=True)
    extra_parser.add_argument('--encoder_path', type=str, nargs='+', required=True)
    extra_parser.add_argument('--ranks', nargs='+', default=['1', '2', '4', '8', '16', '32', '64', 'full'])
    extra_parser.add_argument('--output_dir', type=str, default='analysis/analysis_results/lowrank_eval_flexmoe')
    extra_args, remaining = extra_parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()
    args.num_train_epochs = 0
    # Force fusion_model to flexmoe regardless of what's on the CLI — this script
    # only makes sense for flexmoe checkpoints.
    args.fusion_model = 'flexmoe'

    set_seed(args.seed)
    args.mixed_precision = "fp16" if args.fp16 else "no"
    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=args.cpu)
    device = accelerator.device

    # --- Build modalities (same logic as mimiciv_tasks.py main / fusemoe eval) ---
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
        'ihm-los-pheno-mortality-readmission-birads-risk-density_mod': args.ihm_mod+'_'+args.los_mod+'_'+args.pheno_mod+'_'+args.mor_mod+'_'+args.rad_mod+'_'+args.birads_mod+'_'+args.risk_mod+'_'+args.density_mod,
    }
    task_mod_key = f'{args.task}_mod'  # noqa: F841 (parity with fusemoe script)

    modalities = set()
    modeltype = {}
    for cli_attr, split_key in [
        ('ihm_mod', 'ihm'), ('los_mod', 'los'), ('pheno_mod', 'pheno'),
        ('rad_mod', 'readmission'), ('mor_mod', 'mortality'),
        ('birads_mod', 'birads'), ('risk_mod', 'risk'), ('density_mod', 'density'),
    ]:
        val = getattr(args, cli_attr, '')
        if len(val) != 0 and split_key in args.task.split('-'):
            modeltype[split_key] = '_'.join(sorted(val.split('-')))
            for e in val.split('-'):
                modalities.add(e)

    if 'Text' in modalities:
        BioBert, BioBertConfig, tokenizer = loadBert(args, device)
    else:
        tokenizer, BioBert = None, None

    (all_train, all_valid, all_test, criterion,
     modalities_per_task, train_weights, all_encoders, logits, all_modalities,
    ) = setup_tasks_and_modalities(
        args=args, device=device, tokenizer=tokenizer,
        modeltype=modeltype, modalities=modalities, BioBert=BioBert,
    )

    # --- Load checkpoints ---
    print(f"\nLoading FlexMoE model: {extra_args.model_path}")
    model = torch.load(extra_args.model_path, map_location=device)
    if type(model).__name__ != 'FlexMoE':
        raise TypeError(
            f"Checkpoint class is {type(model).__module__}.{type(model).__name__}, "
            f"not src.flexmoe.FlexMoE. This script only analyzes FlexMoE checkpoints. "
            f"Train with --fusion_model flexmoe first, or use "
            f"src.analysis.eval_lowrank_experts for fusemoe checkpoints."
        )
    model.to_logitslist = model.to_logitslist.to(device)

    task_keys = list(all_encoders.keys())
    encoder_paths = extra_args.encoder_path
    if len(encoder_paths) == 1:
        print(f"Loading shared encoder: {encoder_paths[0]}")
        enc = torch.load(encoder_paths[0], map_location=device).to(device)
        for task_key in task_keys:
            all_encoders[task_key] = enc
    elif len(encoder_paths) == len(task_keys):
        for task_key, enc_path in zip(task_keys, encoder_paths):
            print(f"Loading encoder for {task_key}: {enc_path}")
            all_encoders[task_key] = torch.load(enc_path, map_location=device).to(device)
    else:
        raise ValueError(
            f"Got {len(encoder_paths)} encoder path(s) but {len(task_keys)} tasks ({task_keys})."
        )

    os.makedirs(extra_args.output_dir, exist_ok=True)

    rank_values = []
    for r in extra_args.ranks:
        rank_values.append(None if r.lower() == 'full' else int(r))

    all_results = {}

    # --- Pre-truncation diagnostics ---
    diagnose_expert_contribution(model, all_encoders, all_test,
                                 modalities_per_task, args, device, max_batches=10)

    # --- Router rank analysis ---
    print(f"\n{'='*60}\n  Router Rank Analysis\n{'='*60}")
    router_info = compute_router_ranks(model)
    if router_info:
        for info in router_info:
            print(f"  {info['key']}")
            print(f"    shape={info['shape']}, params={info['params']}")
            print(f"    effective_rank={info['eff_rank']:.2f} / {info['max_rank']}")
            print(f"    rank@90%energy={info['rank_90_energy']}, rank@99%energy={info['rank_99_energy']}")
            print(f"    top_sv={info['top_sv']:.4f}, sv_ratio(1/2)={info['sv_ratio_1_2']:.2f}")
        with open(os.path.join(extra_args.output_dir, 'router_rank_analysis.json'), 'w') as f:
            json.dump(router_info, f, indent=2)
    else:
        print("  No router weight matrices found.")

    # --- Low-rank sweep ---
    for rank in rank_values:
        rank_label = 'full' if rank is None else str(rank)
        print(f"\n{'='*60}\n  Evaluating rank = {rank_label}\n{'='*60}")

        model_copy = copy.deepcopy(model)
        if rank is not None:
            n_modified = apply_lowrank_to_experts(model_copy, rank)
            print(f"  Truncated {n_modified} per-expert weight slices to rank {rank}")
        else:
            print(f"  Using full-rank weights (baseline)")

        full_params, lowrank_params, param_details = count_expert_params(model_copy, rank)
        compression = (1 - lowrank_params / full_params) * 100 if full_params > 0 else 0
        print(f"  Expert params: full={full_params:,}, at_rank={lowrank_params:,} ({compression:.1f}% compression)")
        for key, shape, full_n, factored_n in param_details:
            short_key = key.split('.mlp.experts.')[-1]
            print(f"    experts.{short_key}: {shape} {full_n:,} -> {factored_n:,}")

        results = evaluate(model_copy, all_encoders, all_test,
                           modalities_per_task, args, device)
        all_results[rank_label] = results
        all_results[rank_label]['_param_info'] = {
            'full_expert_params': full_params,
            'lowrank_expert_params': lowrank_params,
            'compression_pct': round(compression, 2),
        }

        for task, metrics in results.items():
            if task.startswith('_'): continue
            mn = metrics.get('metric_name', '?')
            pm = metrics.get('primary_metric', '?')
            extra_line = ''
            if 'f1' in metrics:    extra_line += f", f1={metrics['f1']:.4f}"
            if 'auprc' in metrics: extra_line += f", auprc={metrics['auprc']:.4f}"
            print(f"  {task}: {mn}={pm:.4f}{extra_line}")

        del model_copy
        torch.cuda.empty_cache()

    results_path = os.path.join(extra_args.output_dir, 'lowrank_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Summary ---
    print(f"\n{'='*70}\nSUMMARY: Primary metric by rank\n{'='*70}")
    tasks = [k for k in next(iter(all_results.values())).keys() if not k.startswith('_')]
    header = f"{'rank':>6}  " + "  ".join(f"{t:>14}" for t in tasks) + f"  {'params':>12}  {'compress':>9}"
    print(header)
    for rank_label, res in all_results.items():
        row = f"{rank_label:>6}  "
        for t in tasks:
            pm = res.get(t, {}).get('primary_metric', None)
            row += f"{pm:>14.4f}  " if isinstance(pm, (float, int)) else f"{'-':>14}  "
        info = res.get('_param_info', {})
        row += f"{info.get('lowrank_expert_params', 0):>12,}  {info.get('compression_pct', 0):>8.1f}%"
        print(row)


if __name__ == '__main__':
    main()
