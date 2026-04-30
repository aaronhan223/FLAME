"""Continual learning training loop for clinical-highmmt.

Step-1 skeleton: trains tasks one after another in the order given by
`args.task` (e.g., ``ihm-los-birads`` -> IHM, then LOS, then BIRADS). After
each task, evaluates on every task seen so far. No rank reservation,
expert-pool growth, or parameter freezing yet -- this is the naive
sequential baseline that exposes catastrophic forgetting and verifies the
pipeline plumbing. CL-specific machinery is layered on in later steps.

Reuses ``drop_modalities`` and ``replace_missing_embeddings`` from the
existing multi-task training module so behavior matches one-task-at-a-time.
"""

import copy
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    accuracy_score,
    average_precision_score,
    hamming_loss,
)

from src.train_structure_multitask_mimic import (
    drop_modalities,
    replace_missing_embeddings,
)
from src.eval_scripts.performance import metrics_multilabel, metrics_multiclass
from src.continual.cl_moe import (
    LowRankExpertMLP,
    StackedExpertMLP,
    add_router_head_only,
    append_fresh_active_components,
    convert_to_stacked_after_first_reserve,
    find_seq_moes,
    freeze_router_columns,
    grow_seq_moe,
    reserve_active_components,
    reserve_low_rank,
    set_current_task_idx,
)
from src.continual.cl_routers import (
    ColumnGrowModalityRouter,
    PerTaskModalityRouter,
)
from src.fusemoe_multitask.moe import TemporalExpertMLP

try:
    import wandb
except ImportError:
    wandb = None


_TASK_LABEL_KEY = {'MOR': 'mortality', 'RAD': 'readmission'}


# Mapping from a base modality name (e.g., 'TS', 'CXR') to the attribute
# names on a shared encoder that constitute its sub-encoder. ``ModalityEncoders``
# (used for IHM/LOS/PHENO) is structured per modality this way; see
# ``src/encoders.py``. EMBED-side modalities ('cc', 'mlo', '2dcc', '2dmlo')
# all share a single ``projector`` inside ``EMBEDEncoder`` so per-modality
# splitting isn't meaningful for them; they fall through to whole-encoder
# unfreezing handled below.
_MODALITY_SUBMODULE_MAPPING = {
    'TS': ('time_attn_ts', 'proj_ts', 'moe'),
    'Text': ('bertrep', 'time_attn_text'),
    'CXR': ('time_attn_cxr',),
    'ECG': ('time_attn_ecg',),
}


def _task_modalities(args, task_key):
    """Return the set of base modality names (e.g., ``{'TS', 'CXR'}``) used by
    the task identified by upper-case ``task_key`` (``'IHM'``, ``'LOS'``,
    ``'BIRADS'``, ...). Reads from the corresponding ``--<task>_mod`` arg.
    """
    attr_map = {
        'IHM': 'ihm_mod', 'LOS': 'los_mod', 'PHENO': 'pheno_mod',
        'RAD': 'rad_mod', 'MOR': 'mor_mod',
        'BIRADS': 'birads_mod', 'RISK': 'risk_mod', 'DENSITY': 'density_mod',
    }
    attr = attr_map.get(task_key)
    if attr is None:
        return set()
    s = getattr(args, attr, '') or ''
    return set(s.split('-')) if s else set()


def _unfreeze_first_appearance_submodules(encoder, first_app_modalities):
    """Unfreeze submodules of ``encoder`` that correspond to modalities in
    ``first_app_modalities`` (modalities whose first appearance in the task
    sequence is the current task).

    Three cases:
      * All first-appearing modalities are mapped (e.g., {'CXR'} on
        ``ModalityEncoders``): unfreeze just those submodules. Other modality
        sub-encoders on the same shared instance stay frozen.
      * No first-appearing modalities are mapped (e.g., {'cc','mlo',...} on
        ``EMBEDEncoder``): unfreeze the entire encoder.
      * Mixed: unfreeze the mapped submodules and warn about the unmapped
        ones (so we don't silently over-freeze or over-unfreeze).

    Returns ``(num_params_unfrozen, mode_str, unfrozen_attr_names)``.
    """
    if not first_app_modalities:
        return 0, 'none', []

    target_attrs = []
    unmapped = []
    for mod in first_app_modalities:
        if mod in _MODALITY_SUBMODULE_MAPPING:
            target_attrs.extend(_MODALITY_SUBMODULE_MAPPING[mod])
        else:
            unmapped.append(mod)

    n_unfrozen = 0
    unfrozen_paths = []

    for attr in target_attrs:
        sub = getattr(encoder, attr, None)
        if sub is None:
            continue
        for p in sub.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                n_unfrozen += p.numel()
        unfrozen_paths.append(attr)

    if unmapped and not target_attrs:
        for p in encoder.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                n_unfrozen += p.numel()
        unfrozen_paths.append('<entire encoder>')
        return n_unfrozen, 'whole_encoder', unfrozen_paths
    if unmapped:
        print(f'[CL] warn: first-appearance modalities {unmapped} have no '
              f'granular submodule mapping but other modalities at this task '
              f'do. Their submodules (if any) were not unfrozen; extend '
              f'_MODALITY_SUBMODULE_MAPPING to handle them.')

    return n_unfrozen, ('granular' if target_attrs else 'none'), unfrozen_paths


# Secondary metrics we want to surface alongside the primary in console
# output, ``cl_log``, and the results .txt. Order is the print order; primary
# is shown first, then the rest of these in that order if present.
_REPORT_METRIC_ORDER = (
    # Binary (IHM, LOS, MOR, RAD)
    'auc', 'auprc', 'f1', 'accuracy',
    # Multi-class (BIRADS, DENSITY) and multi-label (PHENO)
    'ave_auc_macro', 'ave_auc_micro', 'ave_auc_weighted', 'auc_mean',
    'ave_auprc_macro', 'ave_auprc_micro',
    'macro_f1', 'micro_f1',
    'subset_accuracy', 'hamming_accuracy',
)


def _fmt_metrics(metrics):
    """One-line summary of a metrics dict for console + log lines.

    Always shows the primary metric first (whatever ``metric_name`` reports),
    then any remaining scalar entries in ``_REPORT_METRIC_ORDER`` that exist
    on the dict. Non-scalar entries (e.g., per-label ``auc_scores`` arrays)
    are skipped.
    """
    primary_name = metrics.get('metric_name')
    primary_val = metrics.get('primary_metric')
    seen = set()
    parts = []
    if primary_name is not None and primary_val is not None:
        parts.append(f'{primary_name}={float(primary_val):.4f}')
        seen.add(primary_name)
    for key in _REPORT_METRIC_ORDER:
        if key in seen or key not in metrics:
            continue
        v = metrics[key]
        if isinstance(v, (int, float, np.floating)):
            parts.append(f'{key}={float(v):.4f}')
            seen.add(key)
    return ' '.join(parts) if parts else 'no metrics'


def _scalar_metric_items(metrics):
    """Yield ``(name, float_value)`` for every scalar entry in metrics
    excluding the meta keys ``primary_metric`` / ``metric_name``."""
    for k, v in metrics.items():
        if k in ('primary_metric', 'metric_name'):
            continue
        if isinstance(v, (int, float, np.floating)):
            yield k, float(v)


def _encode_batch(task, batch, encoder, modalities_t, device):
    """Run the modality encoder for one batch of one task. Returns (embeddings, label)."""
    if task in ['IHM', 'PHENO', 'LOS']:
        (
            ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts,
            input_ids_sequences, attn_mask_sequences, text_emb,
            note_time, note_time_mask, cxr_feats, cxr_time, cxr_time_mask,
            ecg_feats, ecg_time, ecg_time_mask, label,
            cxr_missing, text_missing, ecg_missing,
        ) = batch
        embeddings = encoder(
            x_ts=ts_input_sequences, x_ts_mask=ts_mask_sequences,
            ts_tt_list=ts_tt,
            input_ids_sequences=input_ids_sequences,
            attn_mask_sequences=attn_mask_sequences, text_emb=text_emb,
            note_time_list=note_time, note_time_mask_list=note_time_mask,
            cxr_feats=cxr_feats, cxr_time=cxr_time, cxr_time_mask=cxr_time_mask,
            ecg_feats=ecg_feats, ecg_time=ecg_time, ecg_time_mask=ecg_time_mask,
            labels=label, reg_ts=reg_ts,
            cxr_missing=cxr_missing, text_missing=text_missing, ecg_missing=ecg_missing,
            modalities=modalities_t,
        )
    elif task in ['MOR', 'RAD']:
        label = batch[_TASK_LABEL_KEY[task]].long()
        embeddings = encoder(
            codes=batch['codes'], types=batch['types'],
            timestamps=batch['timestamps'], ages=batch['age'],
            genders=batch['gender'], ethnicities=batch['ethnicity'],
            modalities=modalities_t,
        )
    elif task.lower() in ['birads', 'risk', 'density']:
        _idx, label, embed_2dcc, embed_2dmlo, embed_cc, embed_mlo, all_views = batch
        embeddings = encoder(
            embed_cc=embed_cc, embed_mlo=embed_mlo,
            embed_2dcc=embed_2dcc, embed_2dmlo=embed_2dmlo,
            all_views=all_views, modalities=modalities_t, task=task,
        )
    else:
        raise ValueError(f'Unknown task: {task}')
    return embeddings, label


def _build_indict(embeddings, modalities_t, device, args, missing_embeddings):
    indict = {m: embeddings[m].float().to(device) for m in modalities_t}
    indict, masked_keys = drop_modalities(indict, args.modality_drop_rate)
    if args.modality_drop_rate > 0:
        indict = replace_missing_embeddings(
            indict, missing_embeddings, masked_keys=masked_keys,
        )
    return indict


def _to_logit(out, mod_first):
    if 'PHENO' in mod_first:
        return torch.nn.functional.sigmoid(out)
    if 'birads' in mod_first.lower() or 'density' in mod_first.lower():
        return torch.nn.functional.softmax(out, dim=-1)
    return torch.nn.functional.softmax(out, dim=-1)[:, 1]


def _label_for_loss(label, mod_first, device):
    if 'PHENO' in mod_first:
        return label.float().to(device)
    return label.to(device)


def _compute_metrics(mod_first, all_logits, all_labels):
    """Compute the appropriate metric set for the task type.

    For multi-label PHENO and multi-class BIRADS/DENSITY we report micro
    and macro averages of AUROC, AUPRC, and F1 (plus subset/Hamming
    accuracy for multi-label and standard accuracy for multi-class). This
    mirrors what the multi-task pipeline shows but extends it so all
    averages are visible.
    """
    eval_vals = {}
    if 'PHENO' in mod_first:
        # Multi-label: per-label binary classification, hard threshold 0.5
        # for label-wise predictions.
        all_pred = np.where(all_logits > 0.5, 1, 0)
        eval_vals = metrics_multilabel(all_labels, all_logits, verbose=0)
        # AUPRC averages.
        try:
            eval_vals['ave_auprc_micro'] = float(
                average_precision_score(all_labels, all_logits, average='micro')
            )
            eval_vals['ave_auprc_macro'] = float(
                average_precision_score(all_labels, all_logits, average='macro')
            )
        except Exception as e:
            print(f'[CL] AUPRC compute failed (multi-label): {e}')
        # F1 averages.
        eval_vals['micro_f1'] = float(
            f1_score(all_labels, all_pred, average='micro', zero_division=0)
        )
        eval_vals['macro_f1'] = float(
            f1_score(all_labels, all_pred, average='macro', zero_division=0)
        )
        # Accuracy (subset = exact match across all labels; Hamming = per-label
        # accuracy averaged across labels).
        eval_vals['subset_accuracy'] = float(accuracy_score(all_labels, all_pred))
        eval_vals['hamming_accuracy'] = float(1.0 - hamming_loss(all_labels, all_pred))
        eval_vals['primary_metric'] = float(eval_vals['ave_auc_macro'])
        eval_vals['metric_name'] = 'ave_auc_macro'
    elif 'birads' in mod_first.lower() or 'density' in mod_first.lower():
        # Multi-class: argmax for hard predictions; AUPRC computed one-vs-rest
        # by one-hot encoding the labels.
        eval_vals = metrics_multiclass(all_labels, all_logits, verbose=0)
        all_pred = np.argmax(all_logits, axis=1)
        eval_vals['micro_f1'] = float(
            f1_score(all_labels, all_pred, average='micro', zero_division=0)
        )
        eval_vals['macro_f1'] = float(
            f1_score(all_labels, all_pred, average='macro', zero_division=0)
        )
        eval_vals['accuracy'] = float(accuracy_score(all_labels, all_pred))
        try:
            n_classes = all_logits.shape[1]
            labels_oh = np.eye(n_classes)[all_labels.astype(int)]
            eval_vals['ave_auprc_micro'] = float(
                average_precision_score(labels_oh, all_logits, average='micro')
            )
            eval_vals['ave_auprc_macro'] = float(
                average_precision_score(labels_oh, all_logits, average='macro')
            )
        except Exception as e:
            print(f'[CL] AUPRC compute failed (multi-class): {e}')
        eval_vals['primary_metric'] = float(eval_vals['ave_auc_macro'])
        eval_vals['metric_name'] = 'ave_auc_macro'
    else:
        all_pred = np.where(all_logits > 0.5, 1, 0)
        eval_vals['auc'] = float(roc_auc_score(all_labels, all_logits))
        precisions, recalls, _ = precision_recall_curve(all_labels, all_logits)
        eval_vals['auprc'] = float(auc(recalls, precisions))
        eval_vals['f1'] = float(f1_score(all_labels, all_pred, zero_division=0))
        eval_vals['accuracy'] = float(accuracy_score(all_labels, all_pred))
        eval_vals['primary_metric'] = float(eval_vals['auc'])
        eval_vals['metric_name'] = 'auc'
    return eval_vals


@torch.no_grad()
def evaluate_task(model, encoder, dataloader, modalities_t, task, args, device,
                  missing_embeddings=None, criterion=None, t_idx=None,
                  stage_idx=None):
    """Evaluate a single task's loader. Returns (metrics_dict, avg_loss_or_None).

    ``t_idx`` selects the task-specific logits head ``model.to_logitslist[t_idx]``
    (one per *flat* task in the sequence). ``stage_idx`` selects the routing
    stage (which ``PerTaskModalityRouter`` head fires and how many components
    of each ``StackedExpertMLP`` are summed). For single-task-per-stage runs
    these are equal; for multi-task stages they differ. If ``stage_idx`` is
    omitted, it defaults to ``t_idx`` (preserving the prior pipeline's
    behaviour exactly).
    """
    model.eval()
    encoder.eval()
    if missing_embeddings is None:
        missing_embeddings = torch.nn.ParameterDict()
    if t_idx is not None:
        model.to_logits = model.to_logitslist[t_idx]
    routing_idx = stage_idx if stage_idx is not None else t_idx
    if routing_idx is not None:
        # Routers + stacked-expert components: pick the right *stage* slice.
        set_current_task_idx(model, routing_idx)

    eval_logits, eval_labels = [], []
    loss_sum, loss_steps = 0.0, 0
    for batch in tqdm(dataloader, desc=f'eval {task}', leave=False):
        embeddings, label = _encode_batch(task, batch, encoder, modalities_t, device)
        indict = _build_indict(embeddings, modalities_t, device, args, missing_embeddings)
        out, balance_loss = model(indict, task=task)

        if criterion is not None:
            loss = criterion(out, _label_for_loss(label, modalities_t[0], device))
            if balance_loss is not None:
                loss = loss + args.balance_loss_coef * balance_loss
            loss_sum += loss.item()
            loss_steps += 1

        logit = _to_logit(out, modalities_t[0])
        eval_logits += logit.cpu().numpy().tolist()
        eval_labels += label.cpu().numpy().tolist()

    metrics = _compute_metrics(
        modalities_t[0], np.array(eval_logits), np.array(eval_labels),
    )
    avg_loss = (loss_sum / loss_steps) if loss_steps > 0 else None
    return metrics, avg_loss


def _write_results_txt_stage(savedir_root, s_idx, stage_label, phase,
                              stage_task_indices, task_slugs, cl_log):
    """Append a per-stage block to ``results_<phase>.txt`` under
    ``savedir_root``. Mirrors ``_write_results_txt`` but iterates the stage
    sequence -- one task block per task, grouped under the stage that
    trained it. Header reflects the stage label (e.g. ``ihm-los`` for a
    multi-task stage).
    """
    out_path = os.path.join(savedir_root, f'results_{phase}.txt')
    os.makedirs(savedir_root, exist_ok=True)
    prefix = f'after_stage_{s_idx}/{phase}/test/'
    with open(out_path, 'a') as f:
        f.write('\n')
        f.write('=' * 70 + '\n')
        f.write(f'After stage {s_idx + 1} ({stage_label}) -- {phase} eval\n')
        f.write('=' * 70 + '\n')
        for prior_s in range(s_idx + 1):
            prior_label = '-'.join(
                task_slugs[ii] for ii in stage_task_indices[prior_s]
            )
            f.write(f'\n----- Stage {prior_s} ({prior_label}) -----\n')
            for ii in stage_task_indices[prior_s]:
                s_slug = task_slugs[ii]
                f.write(f'\n------ Task {ii} ({s_slug}) ------\n')
                sub = {
                    k.split('/')[-1]: v
                    for k, v in cl_log.items()
                    if k.startswith(prefix + s_slug + '/')
                }
                shown = set()
                if 'primary' in sub:
                    f.write(f'primary: {sub["primary"]:.6f}\n')
                    shown.add('primary')
                for key in _REPORT_METRIC_ORDER:
                    if key in sub and key not in shown:
                        f.write(f'{key}: {sub[key]:.6f}\n')
                        shown.add(key)
                for key in sorted(sub.keys()):
                    if key not in shown:
                        f.write(f'{key}: {sub[key]:.6f}\n')
                        shown.add(key)


def _write_results_txt(savedir_root, t_idx, slug, phase, task_slugs, task_keys,
                       cl_log):
    """Append a per-task block to ``results_<phase>.txt`` under ``savedir_root``.

    Mirrors the multi-task pipeline's per-task printout in
    ``train_structure_multitask_mimic.train``: one section per evaluated
    task with all scalar metrics keyed under ``after_task_{t}/{phase}/test/{s}/...``.
    Intended to be human-readable; wandb / cl_log are the programmatic surfaces.
    """
    out_path = os.path.join(savedir_root, f'results_{phase}.txt')
    os.makedirs(savedir_root, exist_ok=True)
    prefix = f'after_task_{t_idx}/{phase}/test/'
    with open(out_path, 'a') as f:
        f.write('\n')
        f.write('=' * 70 + '\n')
        f.write(f'After task {t_idx + 1} ({slug}) -- {phase} eval\n')
        f.write('=' * 70 + '\n')
        for s_idx in range(t_idx + 1):
            s_slug = task_slugs[s_idx]
            f.write(f'\n------ Task {s_idx} ({s_slug}) ------\n')
            # Pull every cl_log entry under this task's prefix.
            sub = {
                k.split('/')[-1]: v
                for k, v in cl_log.items()
                if k.startswith(prefix + s_slug + '/')
            }
            # Show primary first, then standard metrics in order, then the rest.
            shown = set()
            if 'primary' in sub:
                f.write(f'primary: {sub["primary"]:.6f}\n')
                shown.add('primary')
            for key in _REPORT_METRIC_ORDER:
                if key in sub and key not in shown:
                    f.write(f'{key}: {sub[key]:.6f}\n')
                    shown.add(key)
            for key in sorted(sub.keys()):
                if key not in shown:
                    f.write(f'{key}: {sub[key]:.6f}\n')
                    shown.add(key)


def _log_trainable_param_breakdown(model, encoders, savedir_root, t_idx, task_slug):
    """Write a per-parameter + per-module breakdown of trainable parameters
    at the start of task ``t_idx`` to
    ``savedir_root/trainable_params_task{t_idx}_{task_slug}.txt``.

    The grand total in the file matches the value ``train_continual`` prints
    just before each task starts (``"trainable params: N (across K tensors)"``).
    Three views are produced, ordered by total size descending:

      * Module summary at depth 1 (top-level module of model + encoder dict).
      * Module summary at depth 2 (e.g., ``model.trans_self_cross_ts_txt.layers``).
      * Module summary at depth 3 (drills into experts / routers / per-task
        projection keys).
      * Per-parameter detail (every trainable tensor with its shape and
        element count).
    """
    from collections import defaultdict

    rows = []  # (full_name, shape, numel)
    for name, p in model.named_parameters():
        if p.requires_grad:
            rows.append((f'model.{name}', tuple(p.shape), p.numel()))
    seen_enc_ids = set()
    for tk, enc in encoders.items():
        # Skip duplicate references when shared encoders alias across tasks.
        eid = id(enc)
        if eid in seen_enc_ids:
            continue
        seen_enc_ids.add(eid)
        for name, p in enc.named_parameters():
            if p.requires_grad:
                rows.append((f'encoders[{tk}].{name}', tuple(p.shape), p.numel()))

    rows.sort(key=lambda r: r[0])
    total = sum(r[2] for r in rows)

    by_depth = defaultdict(lambda: defaultdict(int))
    for name, _, n in rows:
        parts = name.split('.')
        for d in range(1, min(4, len(parts) + 1)):
            prefix = '.'.join(parts[:d])
            by_depth[d][prefix] += n

    os.makedirs(savedir_root, exist_ok=True)
    out_path = os.path.join(
        savedir_root, f'trainable_params_stage{t_idx}_{task_slug}.txt'
    )
    with open(out_path, 'w') as f:
        f.write(f'Trainable parameters at start of stage {t_idx} ({task_slug})\n')
        f.write(f'Grand total: {total:,} across {len(rows)} tensors\n')
        for d in (1, 2, 3):
            f.write('\n' + '=' * 80 + '\n')
            f.write(f'Module summary (depth {d})\n')
            f.write('=' * 80 + '\n')
            f.write(f'{"params":>14}  {"% of total":>10}  module\n')
            f.write('-' * 80 + '\n')
            for prefix, n in sorted(by_depth[d].items(), key=lambda kv: -kv[1]):
                pct = 100.0 * n / max(total, 1)
                f.write(f'{n:>14,}  {pct:>9.2f}%  {prefix}\n')
        f.write('\n' + '=' * 80 + '\n')
        f.write(f'Per-parameter detail ({len(rows)} tensors)\n')
        f.write('=' * 80 + '\n')
        f.write(f'{"params":>14}  {"shape":>22}  name\n')
        f.write('-' * 80 + '\n')
        for name, shape, n in rows:
            f.write(f'{n:>14,}  {str(shape):>22}  {name}\n')
    print(f'[CL] trainable param breakdown -> {out_path}')


def _resolve_num_experts_per_task(args):
    """``--num_of_experts`` is parsed as ``nargs='*'`` so it may be ``int`` or
    ``list``. The continual pipeline uses a uniform per-task pool size."""
    n = args.num_of_experts
    if isinstance(n, (list, tuple)):
        if not n:
            raise ValueError('--num_of_experts is empty')
        return int(n[0])
    return int(n)


def _freeze_all(model, encoders):
    for p in model.parameters():
        p.requires_grad = False
    for enc in encoders.values():
        for p in enc.parameters():
            p.requires_grad = False


def _set_active_for_stage(model, encoders, s_idx, stage_task_indices, task_keys,
                           num_experts_per_task, mode, args):
    """Multi-task variant of ``_set_active_for_task``. Unfreezes the parameters
    that should train for **stage** ``s_idx``, where the stage may contain
    multiple tasks indexed by ``stage_task_indices[s_idx]`` (a list of flat
    task indices into the per-task arrays returned by
    ``setup_tasks_and_modalities``).

    Trainable set:
      - Active expert slot for stage ``s_idx``: in default mode the slot
        range ``[s_idx*E, (s_idx+1)*E)`` of fresh ``TemporalExpertMLP``s; in
        ``--fixed_experts`` mode the *last* component of every
        ``StackedExpertMLP``. **Shared** across all tasks in the stage.
      - For ``per_task_router``: the most-recent (active) router on every
        ``PerTaskModalityRouter``; for ``column_grow``: the entire (grown)
        ``w_gate``/``w_noise`` (the freeze hook does column-level masking).
        **Shared** across all tasks in the stage.
      - For every task ``t`` in the stage: ``to_logitslist[t]`` and
        ``proj1[task]`` / ``proj2[task]`` / ``out_layer[task]`` if those
        ``ModuleDict``s are present.
      - First-appearance encoder submodules: any modality whose first
        appearance in the task sequence is **this stage** (i.e., not used
        in any task of any prior stage).
    """
    cur_indices = stage_task_indices[s_idx]
    fixed_experts = bool(getattr(args, 'fixed_experts', False))

    if fixed_experts:
        for sm in find_seq_moes(model):
            for slot in sm.experts:
                if isinstance(slot, StackedExpertMLP) and len(slot.components) > 0:
                    last = slot.components[-1]
                    if not isinstance(last, LowRankExpertMLP):
                        for p in last.parameters():
                            p.requires_grad = True
                elif isinstance(slot, TemporalExpertMLP):
                    for p in slot.parameters():
                        p.requires_grad = True
            routers = list(sm.routers.values())
            if sm.default_router is not None:
                routers.append(sm.default_router)
            for r in routers:
                if isinstance(r, PerTaskModalityRouter):
                    for p in r.task_routers[-1].parameters():
                        p.requires_grad = True
                else:
                    raise TypeError(
                        f"--fixed_experts requires PerTaskModalityRouter, got "
                        f"{type(r).__name__}."
                    )
    else:
        cur_lo = s_idx * num_experts_per_task
        cur_hi = (s_idx + 1) * num_experts_per_task
        for sm in find_seq_moes(model):
            for i in range(cur_lo, min(cur_hi, len(sm.experts))):
                expert = sm.experts[i]
                if isinstance(expert, LowRankExpertMLP):
                    continue
                for p in expert.parameters():
                    p.requires_grad = True
            routers = list(sm.routers.values())
            if sm.default_router is not None:
                routers.append(sm.default_router)
            for r in routers:
                if isinstance(r, PerTaskModalityRouter):
                    for p in r.task_routers[-1].parameters():
                        p.requires_grad = True
                elif isinstance(r, ColumnGrowModalityRouter):
                    r.w_gate.requires_grad = True
                    r.w_noise.requires_grad = True
                else:
                    for p in r.parameters():
                        p.requires_grad = True

    # Per-task heads + per-task projections for every task in this stage.
    for ii in cur_indices:
        if hasattr(model, 'to_logitslist') and ii < len(model.to_logitslist):
            for p in model.to_logitslist[ii].parameters():
                p.requires_grad = True
        task_key_ii = task_keys[ii]
        for attr in ('proj1', 'proj2', 'out_layer'):
            md = getattr(model, attr, None)
            if md is None:
                continue
            if hasattr(md, 'keys') and task_key_ii in md:
                for p in md[task_key_ii].parameters():
                    p.requires_grad = True

    # First-appearance encoder submodules: union over this stage's tasks
    # minus union over all prior stages' tasks.
    cur_mods = set()
    for ii in cur_indices:
        cur_mods |= _task_modalities(args, task_keys[ii])
    prior_mods = set()
    for prior_s in range(s_idx):
        for ii in stage_task_indices[prior_s]:
            prior_mods |= _task_modalities(args, task_keys[ii])
    first_app = cur_mods - prior_mods

    # Track which encoder instances we've already considered (shared encoders
    # alias across multiple task keys in the dict).
    seen_enc_ids = set()
    for ii in cur_indices:
        enc = encoders[task_keys[ii]]
        eid = id(enc)
        if eid in seen_enc_ids:
            continue
        seen_enc_ids.add(eid)
        n_un, kind, paths = _unfreeze_first_appearance_submodules(enc, first_app)
        if n_un > 0:
            print(f'[CL][stage {s_idx} -> task {task_keys[ii]}] unfroze encoder '
                  f'{kind} submodules for first-appearance modalities '
                  f'{sorted(first_app)}: {n_un:,} params, paths={paths}.')

    if not first_app:
        already = sorted(cur_mods & prior_mods)
        print(f'[CL][stage {s_idx}] kept encoders frozen '
              f'(no first-appearance modalities; already trained: {already}).')


def _set_active_for_task(model, encoders, t_idx, task_key, task_keys,
                         num_experts_per_task, mode, args=None):
    """Unfreeze only the parameters that should train during task ``t_idx``.

    Assumes ``_freeze_all`` was just called. The set of trainable params is:
      - The fresh ``TemporalExpertMLP``s in slots ``[t_idx*E, (t_idx+1)*E)``.
        Earlier slots that are now ``LowRankExpertMLP`` stay frozen.
      - For ``column_grow``: full ``w_gate``/``w_noise`` of every continual
        router (the freeze hook zeros gradients on prior columns at backward).
      - For ``per_task_router``: the most-recent (active) router in each
        ``PerTaskModalityRouter``'s ``task_routers`` list.
      - The current task's logits head ``model.to_logitslist[t_idx]``.
      - The current task's per-task projections ``proj1[task]/proj2[task]/
        out_layer[task]`` if those ``ModuleDict``s are present.
      - ``encoders[task_key]`` *if and only if* this is its first appearance
        in the task sequence (i.e., not shared with any earlier task's
        encoder). This handles the case where a later task brings a fresh
        encoder (e.g., ``EMBEDEncoder`` for BIRADS) that was never trained
        at task 0 because its forward is never invoked there. Shared
        encoders (e.g., ``ModalityEncoders`` aliased across IHM/LOS/PHENO)
        stay frozen to avoid catastrophic forgetting.

    The non-MoE cross-attn backbone (``trans_self_cross_*`` QKV, layernorms,
    residual gates, ...) stays frozen between tasks. The caller is expected
    to leave ``requires_grad`` defaults intact at task 0 (which never goes
    through this branch) so the cross-attn backbone gets trained alongside
    the first task's experts and routers.
    """
    fixed_experts = bool(getattr(args, 'fixed_experts', False)) if args is not None else False

    if fixed_experts:
        # Pool is fixed at E. Each slot is a StackedExpertMLP whose last
        # component is the trainable active expert for the current task.
        # Unfreeze ONLY the last component on every slot.
        for sm in find_seq_moes(model):
            for slot in sm.experts:
                if isinstance(slot, StackedExpertMLP) and len(slot.components) > 0:
                    last = slot.components[-1]
                    if not isinstance(last, LowRankExpertMLP):
                        for p in last.parameters():
                            p.requires_grad = True
                elif isinstance(slot, TemporalExpertMLP):
                    # Pre-first-reservation state (shouldn't occur for t > 0
                    # because task 0 reservation wraps slots in StackedExpertMLP).
                    for p in slot.parameters():
                        p.requires_grad = True

            routers = list(sm.routers.values())
            if sm.default_router is not None:
                routers.append(sm.default_router)
            for r in routers:
                if isinstance(r, PerTaskModalityRouter):
                    for p in r.task_routers[-1].parameters():
                        p.requires_grad = True
                else:
                    raise TypeError(
                        f"--fixed_experts requires PerTaskModalityRouter, got "
                        f"{type(r).__name__}."
                    )
    else:
        cur_lo = t_idx * num_experts_per_task
        cur_hi = (t_idx + 1) * num_experts_per_task

        for sm in find_seq_moes(model):
            for i in range(cur_lo, min(cur_hi, len(sm.experts))):
                expert = sm.experts[i]
                if isinstance(expert, LowRankExpertMLP):
                    continue  # already reserved -> stays frozen
                for p in expert.parameters():
                    p.requires_grad = True

            routers = list(sm.routers.values())
            if sm.default_router is not None:
                routers.append(sm.default_router)
            for r in routers:
                if isinstance(r, PerTaskModalityRouter):
                    for p in r.task_routers[-1].parameters():
                        p.requires_grad = True
                elif isinstance(r, ColumnGrowModalityRouter):
                    r.w_gate.requires_grad = True
                    r.w_noise.requires_grad = True
                else:
                    for p in r.parameters():
                        p.requires_grad = True

    if hasattr(model, 'to_logitslist') and t_idx < len(model.to_logitslist):
        for p in model.to_logitslist[t_idx].parameters():
            p.requires_grad = True

    for attr in ('proj1', 'proj2', 'out_layer'):
        md = getattr(model, attr, None)
        if md is None:
            continue
        if hasattr(md, 'keys') and task_key in md:
            for p in md[task_key].parameters():
                p.requires_grad = True

    # Unfreeze encoder *submodules* corresponding to modalities whose first
    # appearance in the task sequence is the current task. Modality
    # sub-encoders that already trained at an earlier task stay frozen
    # (preventing forgetting). Encoders without a per-modality decomposition
    # (e.g., ``EMBEDEncoder``) are unfrozen wholesale when any of their
    # modalities first appears here. This handles both cases that the
    # previous coarse "unfreeze whole encoder if not aliased" rule missed:
    #   - Shared encoder (e.g., IHM/LOS share ModalityEncoders) where a NEW
    #     modality (CXR at LOS) needs its submodule trained while TS/Text
    #     submodules stay frozen.
    #   - Non-shared encoder (e.g., BIRADS' EMBEDEncoder) where every
    #     modality is fresh and we unfreeze the whole encoder.
    if args is not None:
        cur_encoder = encoders[task_key]
        cur_mods = _task_modalities(args, task_key)
        prior_mods = set()
        for s in range(t_idx):
            prior_mods |= _task_modalities(args, task_keys[s])
        first_app = cur_mods - prior_mods
        n_un, kind, paths = _unfreeze_first_appearance_submodules(
            cur_encoder, first_app
        )
        if n_un > 0:
            print(f'[CL][task {t_idx} ({task_key})] unfroze encoder '
                  f'{kind} submodules for first-appearance modalities '
                  f'{sorted(first_app)}: {n_un:,} params, paths={paths}.')
        else:
            already = sorted(cur_mods & prior_mods)
            print(f'[CL][task {t_idx} ({task_key})] kept encoder frozen '
                  f'(no first-appearance modalities; shared with prior tasks: {already}).')


def _post_task_reserve_freeze_grow(model, t_idx, num_tasks,
                                   num_experts_per_task, rank, mode,
                                   fixed_experts=False):
    """After task ``t_idx`` finishes training: SVD-reserve its experts, freeze
    the corresponding router columns/heads, and (if not the last task)
    prepare the model for task ``t_idx+1``.

    Two structural variants:

    * ``fixed_experts=False`` (default growing-pool mode): SVD-reserve task t's
      experts (slots ``[t*E, (t+1)*E)``), freeze the corresponding router
      heads/columns, then grow the pool by another E experts plus add a new
      router head (sized for the new total).

    * ``fixed_experts=True``: SVD-reserve task t's *active component* on
      every ``StackedExpertMLP`` slot (after the first task this is the slot's
      most-recently-added component; for the very first task we still go
      through ``reserve_low_rank`` and then wrap each slot in a
      ``StackedExpertMLP``). Pool size on the ``SeqMoE`` is unchanged. Router
      head is added per task (per_task_router enforced).
    """
    if fixed_experts:
        for sm in find_seq_moes(model):
            if t_idx == 0:
                # First task: experts are still TemporalExpertMLP. Use the
                # existing reservation path (replaces with LowRankExpertMLP),
                # then wrap each slot in a StackedExpertMLP with one component.
                indices = list(range(len(sm.experts)))
                reserve_low_rank(sm, indices, rank)
                convert_to_stacked_after_first_reserve(sm)
            else:
                # Subsequent tasks: each slot is already a StackedExpertMLP
                # with the just-trained TemporalExpertMLP as its last
                # component. SVD-truncate that last component in place.
                reserve_active_components(sm, rank)
            freeze_router_columns(sm, n_frozen_cols=sm.num_experts, mode=mode)
        print(f'[CL][fixed_experts] reserved task-{t_idx} active component '
              f'(rank {rank}) on every slot; pool size unchanged at '
              f'{next(iter(find_seq_moes(model))).num_experts}.')

        if t_idx < num_tasks - 1:
            for sm in find_seq_moes(model):
                append_fresh_active_components(sm)
                add_router_head_only(sm)
            print(f'[CL][fixed_experts] appended fresh zero-init component '
                  f'to every slot + new router head for next task.')
        return

    # ----- Default growing-pool path (unchanged) -----
    cur_lo = t_idx * num_experts_per_task
    cur_hi = (t_idx + 1) * num_experts_per_task

    for sm in find_seq_moes(model):
        indices = list(range(cur_lo, min(cur_hi, len(sm.experts))))
        replaced = reserve_low_rank(sm, indices, rank)
        freeze_router_columns(sm, n_frozen_cols=cur_hi, mode=mode)
    print(f'[CL] reserved task-{t_idx} experts at rank {rank}; '
          f'froze router heads/columns for slots [0:{cur_hi}].')

    if t_idx < num_tasks - 1:
        for sm in find_seq_moes(model):
            grow_seq_moe(sm, num_experts_per_task)
        new_total = cur_hi + num_experts_per_task
        print(f'[CL] grew expert pool to {new_total} '
              f'(added {num_experts_per_task} fresh experts + new router head).')


def _start_wandb_if_requested(args):
    use_wandb = bool(getattr(args, 'use_wandb', False)) and (wandb is not None)
    started_here = False
    if use_wandb and wandb.run is None:
        run_name = getattr(args, 'wandb_run_name', None) or (
            f'continual_{args.task}_rank{getattr(args, "reserved_rank", "NA")}'
            f'_router_{getattr(args, "router_growth_mode", "NA")}'
            f'{"_fixed_experts" if getattr(args, "fixed_experts", False) else ""}'
            f'_replay{getattr(args, "replay_proportion", 0.0)}'
            f'_lr{args.lr}_wd{args.weight_decay}'
        )
        wandb.init(
            entity='shravan25-jhu',
            project=getattr(args, 'wandb_project', 'clinical-highmmt'),
            name=run_name,
            config=vars(args),
        )
        started_here = True
    return use_wandb, started_here


def train_continual(
    model,
    all_train, all_valid, all_test,
    modalities_per_task,
    criterion,
    train_weights,
    encoders,
    args,
    savedir_root,
    device,
):
    """Stage-based continual training. Each "stage" is one or more tasks
    trained jointly (multi-task) before reservation. Stages run sequentially
    in the order given by the parsed ``args.task_stages`` (set up by
    ``continual_tasks.parse_cl_args``).

    Backward-compat: when ``--task`` has no ``;`` separator, every task is
    its own single-task stage and behavior matches the prior pipeline.
    """
    use_wandb, wandb_started_here = _start_wandb_if_requested(args)

    task_slugs = args.task.split('-')
    task_keys = [modalities_per_task[ii][0].split('_')[1] for ii in range(len(modalities_per_task))]
    num_tasks = len(modalities_per_task)
    num_experts_per_stage = _resolve_num_experts_per_task(args)
    mode = args.router_growth_mode
    rank = int(args.reserved_rank)

    # Stage groupings; default = each task its own stage.
    stage_task_indices = getattr(args, 'task_stage_indices', None)
    stage_slugs_lists = getattr(args, 'task_stages', None)
    if stage_task_indices is None or stage_slugs_lists is None:
        stage_task_indices = [[ii] for ii in range(num_tasks)]
        stage_slugs_lists = [[task_slugs[ii]] for ii in range(num_tasks)]
    num_stages = len(stage_task_indices)

    stage_labels = ['-'.join(stage_slugs) for stage_slugs in stage_slugs_lists]
    print(f'\n[CL] Stage sequence ({num_stages} stages):')
    for s_idx in range(num_stages):
        stage_keys_s = [task_keys[ii] for ii in stage_task_indices[s_idx]]
        print(f'  stage {s_idx}: {stage_labels[s_idx]}  (tasks={stage_keys_s})')
    print(f'[CL] router_growth_mode={mode!r}, reserved_rank={rank}, '
          f'experts_per_stage={num_experts_per_stage}, replay={args.replay_proportion}, '
          f'fixed_experts={bool(getattr(args, "fixed_experts", False))}')

    missing_embeddings = torch.nn.ParameterDict()
    cl_log = {}

    for s_idx in range(num_stages):
        stage_indices = stage_task_indices[s_idx]
        stage_label = stage_labels[s_idx]
        stage_keys = [task_keys[ii] for ii in stage_indices]
        is_multi_task = len(stage_indices) > 1

        print(f'\n{"="*70}\n[CL] Stage {s_idx + 1}/{num_stages}: {stage_label} '
              f'(tasks={stage_keys}, multi_task={is_multi_task})\n{"="*70}')

        # Stage 0 trains the encoders + non-MoE backbone alongside the first
        # stage's experts/routers; later stages freeze everything except the
        # newly-added expert slot, the active router head, and the per-task
        # heads/projections of every task in the stage.
        if s_idx > 0:
            _freeze_all(model, encoders)
            _set_active_for_stage(
                model, encoders, s_idx, stage_task_indices, task_keys,
                num_experts_per_task=num_experts_per_stage, mode=mode, args=args,
            )

        set_current_task_idx(model, s_idx)

        params = [p for p in model.parameters() if p.requires_grad]
        for enc in encoders.values():
            params += [p for p in enc.parameters() if p.requires_grad]
        seen = set()
        params = [p for p in params if id(p) not in seen and not seen.add(id(p))]

        optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        n_train = sum(p.numel() for p in params)
        print(f'[CL][stage {s_idx} ({stage_label})] trainable params: {n_train:,} '
              f'(across {len(params)} tensors)')
        _log_trainable_param_breakdown(model, encoders, savedir_root, s_idx, stage_label)

        savedir_stage = os.path.join(savedir_root, f'stage{s_idx}_{stage_label}.pt')
        os.makedirs(os.path.dirname(savedir_stage), exist_ok=True)

        # Build the multi-task batch structure for this stage. Each entry of
        # ``fulltrains`` is a dict keyed by the *flat* task index, mapping to
        # one batch from that task's loader. Tasks with fewer batches stop
        # contributing once exhausted (matches the existing multi-task code).
        fulltrains = []
        for ii in stage_indices:
            for batch_pos, batch in enumerate(all_train[ii]):
                if batch_pos >= len(fulltrains):
                    fulltrains.append({})
                fulltrains[batch_pos][ii] = batch
        if not fulltrains:
            print(f'[CL][stage {s_idx}] WARNING: no training batches; skipping.')
            continue

        best_score = -float('inf')
        best_model_state = None
        best_encoder_states = {}

        for ep in range(args.num_train_epochs):
            model.train()
            seen_enc_ids = set()
            for ii in stage_indices:
                enc = encoders[task_keys[ii]]
                if id(enc) not in seen_enc_ids:
                    seen_enc_ids.add(id(enc))
                    enc.train()
            print(f'\n[CL][stage {s_idx}] Train epoch {ep + 1}/{args.num_train_epochs}...')
            ep_loss_sum, ep_steps = 0.0, 0

            for js in tqdm(fulltrains, desc=f'train stage{s_idx}'):
                optim.zero_grad()
                losses = 0.0
                for ii in js:
                    task = task_keys[ii]
                    modalities_t = modalities_per_task[ii]
                    encoder_t = encoders[task]
                    crit_t = criterion[ii]

                    model.to_logits = model.to_logitslist[ii]

                    embeddings, label = _encode_batch(task, js[ii], encoder_t, modalities_t, device)
                    indict = _build_indict(embeddings, modalities_t, device, args, missing_embeddings)

                    tracked = {id(p) for grp in optim.param_groups for p in grp['params']}
                    fresh_missing = [p for p in missing_embeddings.parameters() if id(p) not in tracked]
                    if fresh_missing:
                        optim.add_param_group({'params': fresh_missing})

                    out, balance_loss = model(indict, task=task)
                    loss = crit_t(out, _label_for_loss(label, modalities_t[0], device))
                    if balance_loss is not None:
                        loss = loss + args.balance_loss_coef * balance_loss
                    loss = loss * train_weights[ii]
                    losses = losses + loss

                if isinstance(losses, torch.Tensor):
                    losses.backward()
                    optim.step()
                    ep_loss_sum += losses.item()
                    ep_steps += 1

            train_log = {
                f'train/stage{s_idx}/loss': ep_loss_sum / max(ep_steps, 1),
                'cl/stage_idx': s_idx,
                'epoch': ep,
            }

            stage_val_sum = 0.0
            val_log = {}
            for ii in stage_indices:
                task = task_keys[ii]
                slug = task_slugs[ii]
                val_metrics, val_loss = evaluate_task(
                    model, encoders[task], all_valid[ii],
                    modalities_per_task[ii], task, args, device,
                    missing_embeddings=missing_embeddings,
                    criterion=criterion[ii], t_idx=ii, stage_idx=s_idx,
                )
                primary = val_metrics.get('primary_metric', 0.0)
                stage_val_sum += primary
                print(f'[CL][stage {s_idx}][{slug}] val {_fmt_metrics(val_metrics)}')
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float, np.floating)):
                        val_log[f'val/stage{s_idx}/{slug}/{k}'] = float(v)
                if val_loss is not None:
                    val_log[f'val/stage{s_idx}/{slug}/loss'] = val_loss

            print(f'[CL][stage {s_idx}] sum of val primary metrics = {stage_val_sum:.4f}')

            if stage_val_sum > best_score:
                best_score = stage_val_sum
                torch.save(model, savedir_stage)
                best_model_state = copy.deepcopy(model.state_dict())
                snap_seen = set()
                best_encoder_states = {}
                for ii in stage_indices:
                    task = task_keys[ii]
                    enc = encoders[task]
                    if id(enc) in snap_seen:
                        continue
                    snap_seen.add(id(enc))
                    torch.save(enc, savedir_stage.replace('.pt', f'_{task}_encoder.pt'))
                    best_encoder_states[task] = copy.deepcopy(enc.state_dict())
                print(f'[CL][stage {s_idx}] saved best at val_sum={stage_val_sum:.4f}')

            if use_wandb:
                wandb.log({**train_log, **val_log})

        # Restore best-on-val snapshot before reservation.
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            for task, enc_state in best_encoder_states.items():
                encoders[task].load_state_dict(enc_state)
            print(f'[CL][stage {s_idx}] restored best-on-val snapshot '
                  f'(val_sum={best_score:.4f}) for full-rank eval + reservation.')

        # --- (1) Full-rank after-stage eval: every task in every stage 0..s_idx.
        print(f'\n[CL] After stage {s_idx + 1} ({stage_label}) full-rank eval')
        full_rank_after_log = {}
        for prior_s in range(s_idx + 1):
            for ii in stage_task_indices[prior_s]:
                s_task = task_keys[ii]
                s_slug = task_slugs[ii]
                test_metrics, _ = evaluate_task(
                    model, encoders[s_task], all_test[ii],
                    modalities_per_task[ii], s_task, args, device,
                    missing_embeddings=missing_embeddings,
                    t_idx=ii, stage_idx=prior_s,
                )
                primary = test_metrics.get('primary_metric', 0.0)
                print(f'[CL][after-stage{s_idx}][full-rank] test on {s_slug} '
                      f'(stage {prior_s}): {_fmt_metrics(test_metrics)}')
                cl_log[f'after_stage_{s_idx}/full_rank/test/{s_slug}/primary'] = float(primary)
                for k, v in _scalar_metric_items(test_metrics):
                    cl_log[f'after_stage_{s_idx}/full_rank/test/{s_slug}/{k}'] = v
                    full_rank_after_log[f'cl/after_stage_{s_idx}/full_rank/test/{s_slug}/{k}'] = v
        if use_wandb and full_rank_after_log:
            wandb.log(full_rank_after_log)
        _write_results_txt_stage(
            savedir_root, s_idx, stage_label, phase='full_rank',
            stage_task_indices=stage_task_indices, task_slugs=task_slugs,
            cl_log=cl_log,
        )

        # --- (2) Post-stage: SVD-reserve, freeze, grow.
        _post_task_reserve_freeze_grow(
            model, s_idx, num_tasks=num_stages,
            num_experts_per_task=num_experts_per_stage, rank=rank, mode=mode,
            fixed_experts=bool(getattr(args, 'fixed_experts', False)),
        )

        post_path = os.path.join(savedir_root, f'stage{s_idx}_{stage_label}_reserved.pt')
        torch.save(model, post_path)

        # --- (3) Reserved-rank after-stage eval.
        print(f'\n[CL] After stage {s_idx + 1} ({stage_label}) reserved-rank eval')
        after_stage_log = {}
        for prior_s in range(s_idx + 1):
            for ii in stage_task_indices[prior_s]:
                s_task = task_keys[ii]
                s_slug = task_slugs[ii]
                test_metrics, _ = evaluate_task(
                    model, encoders[s_task], all_test[ii],
                    modalities_per_task[ii], s_task, args, device,
                    missing_embeddings=missing_embeddings,
                    t_idx=ii, stage_idx=prior_s,
                )
                primary = test_metrics.get('primary_metric', 0.0)
                print(f'[CL][after-stage{s_idx}][reserved] test on {s_slug} '
                      f'(stage {prior_s}): {_fmt_metrics(test_metrics)}')
                cl_log[f'after_stage_{s_idx}/reserved/test/{s_slug}/primary'] = float(primary)
                # Legacy per-task keys for back-compat with any consumer that
                # parsed the old single-task-per-stage results.
                cl_log[f'after_task_{s_idx}/test/{s_slug}/primary'] = float(primary)
                for k, v in _scalar_metric_items(test_metrics):
                    cl_log[f'after_stage_{s_idx}/reserved/test/{s_slug}/{k}'] = v
                    cl_log[f'after_task_{s_idx}/test/{s_slug}/{k}'] = v
                    after_stage_log[f'cl/after_stage_{s_idx}/reserved/test/{s_slug}/{k}'] = v
        if use_wandb and after_stage_log:
            wandb.log(after_stage_log)
        _write_results_txt_stage(
            savedir_root, s_idx, stage_label, phase='reserved',
            stage_task_indices=stage_task_indices, task_slugs=task_slugs,
            cl_log=cl_log,
        )

    if use_wandb and wandb_started_here:
        wandb.finish()
    return cl_log


def _legacy_train_continual_unused(  # noqa: kept until verified the new impl works
    model, all_train, all_valid, all_test, modalities_per_task,
    criterion, train_weights, encoders, args, savedir_root, device,
):
    """The single-task-per-stage version, kept as a reference. Not invoked."""
    use_wandb, wandb_started_here = _start_wandb_if_requested(args)

    task_slugs = args.task.split('-')
    task_keys = [modalities_per_task[ii][0].split('_')[1] for ii in range(len(modalities_per_task))]
    num_tasks = len(modalities_per_task)
    num_experts_per_task = _resolve_num_experts_per_task(args)
    mode = args.router_growth_mode
    rank = int(args.reserved_rank)

    missing_embeddings = torch.nn.ParameterDict()
    cl_log = {}

    for t_idx in range(num_tasks):
        task = task_keys[t_idx]
        slug = task_slugs[t_idx]
        train_loader = all_train[t_idx]
        valid_loader = all_valid[t_idx]
        modalities_t = modalities_per_task[t_idx]
        encoder_t = encoders[task]
        crit_t = criterion[t_idx]

        print(f'\n{"="*70}\n[CL] Task {t_idx + 1}/{num_tasks}: {slug} ({task})\n{"="*70}')

        # Task 0 trains the encoders + non-MoE backbone alongside the first
        # task's experts/routers; later tasks freeze everything except the
        # newly-added expert slots, the active router head, the current task's
        # logits head, and the current task's per-task projections.
        if t_idx > 0:
            _freeze_all(model, encoders)
            _set_active_for_task(
                model, encoders, t_idx, task, task_keys,
                num_experts_per_task=num_experts_per_task, mode=mode, args=args,
            )

        # For per_task_router: tell each PerTaskModalityRouter which head fires.
        # No-op for column_grow.
        set_current_task_idx(model, t_idx)

        params = [p for p in model.parameters() if p.requires_grad]
        for enc in encoders.values():
            params += [p for p in enc.parameters() if p.requires_grad]
        # Drop duplicates (shared encoder modules show up in multiple dict keys).
        seen = set()
        params = [p for p in params if id(p) not in seen and not seen.add(id(p))]

        optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        n_train = sum(p.numel() for p in params)
        print(f'[CL][{slug}] trainable params: {n_train:,} (across '
              f'{len(params)} tensors)')
        _log_trainable_param_breakdown(model, encoders, savedir_root, t_idx, slug)

        savedir_task = os.path.join(savedir_root, f'task{t_idx}_{slug}.pt')
        os.makedirs(os.path.dirname(savedir_task), exist_ok=True)

        best_score = -float('inf')
        best_model_state = None
        best_encoder_state = None
        for ep in range(args.num_train_epochs):
            model.train()
            encoder_t.train()
            print(f'\n[CL][{slug}] Train epoch {ep + 1}/{args.num_train_epochs}...')
            ep_loss_sum, ep_steps = 0.0, 0

            for batch in tqdm(train_loader, desc=f'train {task}'):
                optim.zero_grad()
                model.to_logits = model.to_logitslist[t_idx]

                embeddings, label = _encode_batch(task, batch, encoder_t, modalities_t, device)
                indict = _build_indict(embeddings, modalities_t, device, args, missing_embeddings)

                # When modality_drop_rate > 0, replace_missing_embeddings may add
                # new params to ``missing_embeddings`` mid-epoch. Make sure the
                # optimizer tracks them.
                tracked = {id(p) for grp in optim.param_groups for p in grp['params']}
                fresh_missing = [p for p in missing_embeddings.parameters() if id(p) not in tracked]
                if fresh_missing:
                    optim.add_param_group({'params': fresh_missing})

                out, balance_loss = model(indict, task=task)
                loss = crit_t(out, _label_for_loss(label, modalities_t[0], device))
                if balance_loss is not None:
                    loss = loss + args.balance_loss_coef * balance_loss
                loss = loss * train_weights[t_idx]
                loss.backward()
                optim.step()

                ep_loss_sum += loss.item()
                ep_steps += 1

            train_log = {
                f'train/{slug}/loss': ep_loss_sum / max(ep_steps, 1),
                'cl/task_idx': t_idx,
                'epoch': ep,
            }

            val_metrics, val_loss = evaluate_task(
                model, encoder_t, valid_loader, modalities_t, task, args, device,
                missing_embeddings=missing_embeddings, criterion=crit_t, t_idx=t_idx,
            )
            primary = val_metrics.get('primary_metric', 0.0)
            val_log = {
                f'val/{slug}/{k}': float(v)
                for k, v in val_metrics.items()
                if isinstance(v, (int, float, np.floating))
            }
            if val_loss is not None:
                val_log[f'val/{slug}/loss'] = val_loss
            print(f'[CL][{slug}] val {_fmt_metrics(val_metrics)}')

            if primary > best_score:
                best_score = primary
                torch.save(model, savedir_task)
                torch.save(
                    encoder_t,
                    savedir_task.replace('.pt', f'_{task}_encoder.pt'),
                )
                # Keep an in-memory snapshot so we can restore the best state
                # before running the full-rank after-task eval and before
                # applying SVD reservation. ``deepcopy`` of the state_dict
                # captures parameter values without aliasing the live tensors.
                best_model_state = copy.deepcopy(model.state_dict())
                best_encoder_state = copy.deepcopy(encoder_t.state_dict())
                print(f'[CL][{slug}] saved best model at val={primary:.4f}')

            if use_wandb:
                wandb.log({**train_log, **val_log})

        # --- Restore best-on-val snapshot before the after-task eval +
        # reservation. Without this restore, the reservation would operate on
        # the last-epoch model, which can be worse than the best-on-val
        # checkpoint. Restoring keeps the CL flow grounded in the best state.
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            encoder_t.load_state_dict(best_encoder_state)
            print(f'[CL][{slug}] restored best-on-val snapshot '
                  f'(val_primary={best_score:.4f}) for full-rank eval + reservation.')

        # --- (1) Full-rank after-task eval: tasks 0..t evaluated on the best
        # model BEFORE the just-trained task's experts are SVD-truncated.
        # Note: for t > 0, prior tasks' experts are already LowRankExpertMLP
        # (frozen at their reservation rank). So "full-rank" here means
        # "current task's experts are still TemporalExpertMLP".
        print(f'\n[CL] After task {t_idx + 1} ({slug}) full-rank eval -> tasks 0..{t_idx}')
        full_rank_after_log = {}
        for s_idx in range(t_idx + 1):
            s_task = task_keys[s_idx]
            s_slug = task_slugs[s_idx]
            set_current_task_idx(model, s_idx)
            test_metrics, _ = evaluate_task(
                model, encoders[s_task], all_test[s_idx],
                modalities_per_task[s_idx], s_task, args, device,
                missing_embeddings=missing_embeddings, t_idx=s_idx,
            )
            primary = test_metrics.get('primary_metric', 0.0)
            print(f'[CL][after-{slug}][full-rank] test on {s_slug}: '
                  f'{_fmt_metrics(test_metrics)}')
            cl_log[f'after_task_{t_idx}/full_rank/test/{s_slug}/primary'] = float(primary)
            for k, v in _scalar_metric_items(test_metrics):
                cl_log[f'after_task_{t_idx}/full_rank/test/{s_slug}/{k}'] = v
                full_rank_after_log[f'cl/after_task_{t_idx}/full_rank/test/{s_slug}/{k}'] = v
        if use_wandb and full_rank_after_log:
            wandb.log(full_rank_after_log)
        _write_results_txt(
            savedir_root, t_idx, slug, phase='full_rank',
            task_slugs=task_slugs, task_keys=task_keys,
            cl_log=cl_log,
        )

        # --- (2) Post-task: SVD-reserve task t's experts at args.reserved_rank,
        # freeze the corresponding router heads/columns, and grow the pool +
        # routers in preparation for task t+1. Operates on the best-on-val
        # state we just restored above.
        _post_task_reserve_freeze_grow(
            model, t_idx, num_tasks=num_tasks,
            num_experts_per_task=num_experts_per_task, rank=rank, mode=mode,
            fixed_experts=bool(getattr(args, 'fixed_experts', False)),
        )

        # Persist the post-reservation model snapshot so reloading reproduces
        # the rank-r state (the best-on-val checkpoint above is the pre-reserve
        # full-rank version; this one is what task t+1 actually starts from).
        post_path = os.path.join(savedir_root, f'task{t_idx}_{slug}_reserved.pt')
        torch.save(model, post_path)

        # --- (3) Reserved-rank after-task eval: tasks 0..t with the
        # rank-``args.reserved_rank`` model. This is the state the model
        # carries forward to task t+1.
        print(f'\n[CL] After task {t_idx + 1} ({slug}) reserved-rank eval -> tasks 0..{t_idx}')
        after_task_log = {}
        for s_idx in range(t_idx + 1):
            s_task = task_keys[s_idx]
            s_slug = task_slugs[s_idx]
            set_current_task_idx(model, s_idx)
            test_metrics, _ = evaluate_task(
                model, encoders[s_task], all_test[s_idx],
                modalities_per_task[s_idx], s_task, args, device,
                missing_embeddings=missing_embeddings, t_idx=s_idx,
            )
            primary = test_metrics.get('primary_metric', 0.0)
            print(f'[CL][after-{slug}][reserved] test on {s_slug}: '
                  f'{_fmt_metrics(test_metrics)}')
            cl_log[f'after_task_{t_idx}/reserved/test/{s_slug}/primary'] = float(primary)
            # Keep the legacy unprefixed key for backward compatibility with
            # any downstream consumers expecting ``after_task_{t}/test/...``.
            cl_log[f'after_task_{t_idx}/test/{s_slug}/primary'] = float(primary)
            for k, v in _scalar_metric_items(test_metrics):
                cl_log[f'after_task_{t_idx}/reserved/test/{s_slug}/{k}'] = v
                cl_log[f'after_task_{t_idx}/test/{s_slug}/{k}'] = v
                after_task_log[f'cl/after_task_{t_idx}/reserved/test/{s_slug}/{k}'] = v
        if use_wandb and after_task_log:
            wandb.log(after_task_log)
        _write_results_txt(
            savedir_root, t_idx, slug, phase='reserved',
            task_slugs=task_slugs, task_keys=task_keys,
            cl_log=cl_log,
        )

    if use_wandb and wandb_started_here:
        wandb.finish()
    return cl_log
