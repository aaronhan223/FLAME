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
    LoRAAdapter,
    LoRAExpertMLP,
    LowRankExpertMLP,
    StackedExpertMLP,
    add_router_head_only,
    append_fresh_active_components,
    append_lora_adapter,
    convert_to_lora_after_stage_0,
    convert_to_stacked_after_first_reserve,
    find_seq_moes,
    freeze_active_lora_adapter,
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
from src.continual.cl_stages import TASK_KEY_TO_SLUG
from src.continual.cl_ewc import (
    EWCState,
    iter_expert_and_router_weights,
    iter_expert_weights,
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
    elif task.lower() == 'diag':
        # ADNI multi-modal diagnosis: batch is ``(idx, label, mod_tensors)``
        # where mod_tensors is a dict modality_name -> tensor. Matches the
        # multi-task pipeline's branch in
        # ``train_structure_multitask_mimic._run_test_loop``.
        _idx, label, mod_tensors = batch
        embeddings = encoder(
            mod_tensors=mod_tensors, modalities=modalities_t, task=task,
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
    if ('birads' in mod_first.lower() or 'density' in mod_first.lower() or 'diag' in mod_first.lower()):
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
    elif ('birads' in mod_first.lower() or 'density' in mod_first.lower() or 'diag' in mod_first.lower()):
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


_TASK_KEY_TO_MOD_ARG = {
    'IHM': 'ihm_mod', 'LOS': 'los_mod', 'PHENO': 'pheno_mod',
    'RAD': 'rad_mod', 'MOR': 'mor_mod',
    'BIRADS': 'birads_mod', 'RISK': 'risk_mod', 'DENSITY': 'density_mod',
}


def _path_safe_task_arg(task_str):
    """Replace ``;`` with ``__`` for filesystem-safe paths. Mirrors the helper
    in ``continual_tasks`` -- duplicated here to avoid the import cycle."""
    return task_str.replace(';', '__')


def _write_best_model_results_for_stage(args, s_idx, stage_label,
                                         stage_task_indices, task_keys,
                                         task_slugs, cl_log):
    """Write a ``best_model_results_*.txt`` file per CL stage in the same
    format as the multi-task pipeline's
    ``train_structure_multitask_mimic._run_test_loop(..., result_filename_prefix='best_model_results')``.

    The file is written under ``{args.results_dir}/{fusion_model}/continual_{cl_method}/...``
    so each baseline lands in its own subtree (and ``aggregate_results.py``
    can scan ``continual_<method>`` subtrees the same way it scans
    ``singletask``/``multitask``).

    Each call appends a fresh per-stage section to the file. The section
    contains one ``------Task {ii} ({slug}, stage {prior_s})------`` block
    per task in stages ``0..s_idx``, using the *full-rank* (best-on-val)
    after-stage eval values from ``cl_log``.
    """
    cl_method = getattr(args, 'cl_method', 'ours')
    fusion = getattr(args, 'fusion_model', 'fusemoe')
    gating = args.gating_function[0] if args.gating_function else 'default'
    task_raw_safe = _path_safe_task_arg(getattr(args, 'task_raw', args.task))
    seed = args.seed
    n_experts = args.num_of_experts[0] if isinstance(args.num_of_experts, (list, tuple)) else args.num_of_experts
    mod_drop_rate = args.modality_drop_rate

    # Build a task-mods string for the filename: hyphenated mods of the
    # tasks in *this* stage, joined with '_'. Matches the multitask
    # pipeline's filename style ('TS-Text' for IHM, 'TS-CXR' for LOS, etc.).
    stage_task_mods = []
    for ii in stage_task_indices[s_idx]:
        task_key = task_keys[ii]
        attr = _TASK_KEY_TO_MOD_ARG.get(task_key)
        if attr is None:
            stage_task_mods.append(task_slugs[ii])
        else:
            stage_task_mods.append(getattr(args, attr, '') or task_slugs[ii])
    task_mods_str = '_'.join(stage_task_mods)

    # Path layout puts the CL stage *above* the seed directory so each
    # ``stage{s}_<label>/<seed>/`` contains exactly one
    # ``best_model_results_*.txt`` file. ``aggregate_results.py`` can then be
    # pointed at ``.../experts_E/stage{s}_<label>/`` and aggregate that
    # specific stage across seeds the same way it does for singletask runs.
    heads_only = bool(getattr(args, 'heads_only', False))
    cl_dir = f'continual_{cl_method}' + ('_heads_only' if heads_only else '')
    out_dir = os.path.join(
        getattr(args, 'results_dir', './results'),
        fusion,
        cl_dir,
        gating,
        task_raw_safe,
        f'mod_drop_rate_{mod_drop_rate}',
        f'experts_{n_experts}',
        f'stage{s_idx}_{stage_label}',
        str(seed),
    )
    out_fname = os.path.join(
        out_dir,
        f'best_model_results_{stage_label}_{task_mods_str}_'
        f'lr{args.lr}_wd{args.weight_decay}_mod_drop_rate_{mod_drop_rate}.txt',
    )
    os.makedirs(out_dir, exist_ok=True)

    # Source-phase rules per CL method:
    #   ours -> reserved (rank-truncated; that's what carries forward).
    #   lora -> full_rank (no truncation; full_rank == reserved here).
    #   ewc  -> full_rank (no truncation; experts continually fine-tuned).
    # heads_only ablation overrides the cl_method's reservation and uses
    # full_rank metrics regardless (no SVD step ever runs).
    if heads_only:
        phase = 'full_rank'
    else:
        phase = 'reserved' if cl_method == 'ours' else 'full_rank'

    setting_line = (
        f'cl_method={cl_method} phase={phase} task={getattr(args, "task_raw", args.task)} '
        f'stage={s_idx}/{stage_label} seed={seed} '
        f'lr={args.lr} wd={args.weight_decay} '
        f'experts={n_experts} reserved_rank={args.reserved_rank} '
        f'lora_rank={getattr(args, "lora_cl_rank", "NA")} '
        f'router={args.router_growth_mode} fixed_experts={getattr(args, "fixed_experts", False)} '
        f'replay={args.replay_proportion} alpha={args.alpha} '
        f'mod_drop_rate={mod_drop_rate}'
    )

    prefix = f'after_stage_{s_idx}/{phase}/test/'
    # Use the exact header string ``aggregate_results.py`` matches against
    # so its existing regex picks our blocks up without modification.
    header_label = 'Final Best Model Test'
    with open(out_fname, 'a') as f:
        f.write(f"\n################## {header_label} ##################\n")
        f.write(setting_line + '  \n')
        for prior_s in range(s_idx + 1):
            for ii in stage_task_indices[prior_s]:
                slug = task_slugs[ii]
                # Match multi-task pipeline format exactly:
                # ``------Task {idx}------``. Stage info lives in the
                # setting line and the directory path.
                f.write(f'------Task {ii}------\n')
                # Human-readable annotation lines. Values are intentionally
                # non-numeric (``stage_idx: stage{N}``) so the aggregator's
                # ``float()`` parse fails and silently skips them, keeping
                # the format compatible with ``aggregate_results.py``.
                f.write(f'task_name: {slug}\n')
                f.write('\n')
                f.write(f'stage_idx: stage{prior_s}\n')
                f.write('\n')
                sub = {
                    k.split('/')[-1]: v
                    for k, v in cl_log.items()
                    if k.startswith(prefix + slug + '/')
                }
                shown = set()
                if 'primary' in sub:
                    f.write(f'primary: {sub["primary"]}\n')
                    f.write('\n')
                    shown.add('primary')
                for key in _REPORT_METRIC_ORDER:
                    if key in sub and key not in shown:
                        f.write(f'{key}: {sub[key]}\n')
                        f.write('\n')
                        shown.add(key)
                for key in sorted(sub.keys()):
                    if key not in shown:
                        f.write(f'{key}: {sub[key]}\n')
                        f.write('\n')
                        shown.add(key)
    print(f'[CL] wrote best_model_results -> {out_fname}')
    return out_fname


def _print_and_write_progression_table(args, savedir_root, num_stages,
                                        stage_task_indices, task_keys,
                                        task_slugs, cl_log):
    """At the end of all stages, print and persist a single summary table:
    rows = tasks (in user's stage order), columns = stages, cells = the
    primary metric for that task at the end of each stage.

    Cells are ``--`` for stages where the task hadn't been trained yet
    (i.e., stage < first_stage(task)). For stages >= first_stage(task),
    the cell shows the primary metric reported by the after-stage eval.

    Also computes:
      * **Final-stage average** -- mean of primary metrics on the last row
        across tasks.
      * **Backward Transfer (BWT)** -- per task, ``primary[final] -
        primary[task's first stage]``. Negative = forgetting. Average BWT
        is the standard CL forgetting metric.
    """
    cl_method = getattr(args, 'cl_method', 'ours')
    if getattr(args, 'heads_only', False):
        phase = 'full_rank'
    else:
        phase = 'reserved' if cl_method == 'ours' else 'full_rank'

    # Tasks in user's stage order: each task's "first stage" is the stage
    # at which it was first trained. Multi-task stages contribute multiple
    # task rows with the same first-stage index.
    rows = []
    for s_idx in range(num_stages):
        for ii in stage_task_indices[s_idx]:
            slug = task_slugs[ii]
            key = task_keys[ii]
            primary_per_stage = []
            for s in range(num_stages):
                v = cl_log.get(
                    f'after_stage_{s}/{phase}/test/{slug}/primary', None
                )
                primary_per_stage.append(v)
            rows.append({
                'slug': slug, 'key': key, 'first_stage': s_idx,
                'metrics': primary_per_stage,
            })

    if not rows:
        return

    # Column widths.
    name_w = max(8, max(len(r['slug']) for r in rows))
    stage_w = 9

    header_cells = (['Task'.ljust(name_w), 'FirstStg'.rjust(8)]
                    + [f'St{s}'.rjust(stage_w) for s in range(num_stages)]
                    + ['BWT'.rjust(stage_w)])
    header = ' | '.join(header_cells)
    sep = '-' * len(header)

    out_lines = []

    def _emit(line):
        print(line)
        out_lines.append(line)

    title_w = max(len(header), 70)
    _emit('')
    _emit('=' * title_w)
    _emit(' Continual learning progression -- primary metric per task per stage'.ljust(title_w))
    _emit(f' cl_method={cl_method}, phase={phase}, num_stages={num_stages}'.ljust(title_w))
    _emit('=' * title_w)
    _emit(header)
    _emit(sep)

    bwts = []
    for row in rows:
        first = row['first_stage']
        first_metric = row['metrics'][first]
        final_metric = row['metrics'][num_stages - 1]
        if first_metric is not None and final_metric is not None:
            bwt = final_metric - first_metric
            bwts.append(bwt)
        else:
            bwt = None
        cells = [row['slug'].ljust(name_w), str(first).rjust(8)]
        for s in range(num_stages):
            v = row['metrics'][s]
            if v is None or s < first:
                cells.append('--'.rjust(stage_w))
            else:
                cells.append(f'{v:.4f}'.rjust(stage_w))
        if bwt is None:
            cells.append('--'.rjust(stage_w))
        else:
            sign = '+' if bwt >= 0 else ''
            cells.append(f'{sign}{bwt:.4f}'.rjust(stage_w))
        _emit(' | '.join(cells))
    _emit(sep)

    # Aggregate metrics: final-stage average, average BWT.
    final_vals = [r['metrics'][num_stages - 1]
                  for r in rows if r['metrics'][num_stages - 1] is not None]
    avg_final = sum(final_vals) / max(len(final_vals), 1) if final_vals else float('nan')
    avg_bwt = sum(bwts) / max(len(bwts), 1) if bwts else float('nan')
    _emit(f' Average final-stage primary metric: {avg_final:.4f}')
    _emit(f' Average BWT (final - first per task): {avg_bwt:+.4f}'
          + ('   (negative = forgetting)' if not (avg_bwt != avg_bwt) else ''))
    _emit('=' * title_w)

    # Persist.
    progression_path = os.path.join(savedir_root, 'progression_summary.txt')
    with open(progression_path, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')
    print(f'[CL] progression summary written to: {progression_path}')


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


def _ewc_pseudo_label(out, mod_first, sampling, true_label, device):
    """Construct the target tensor used for the Fisher-information backward
    pass under each sampling mode supported by the reference EWC.

      * ``true``: use the actual training label for this task.
      * ``max_pred``: model's argmax (binary/multiclass) or threshold-at-0.5
        (multilabel) of its current output. Lets Fisher reflect what the
        model has *learned* rather than the supervision signal.
      * ``multinomial``: binary/multiclass: sample a class index from
        ``softmax(out)``. Multilabel: bernoulli-sample from ``sigmoid(out)``.
    """
    if sampling == 'true':
        return _label_for_loss(true_label, mod_first, device)

    detached = out.detach()
    if 'PHENO' in mod_first:
        # Multi-label: per-label sigmoid -> bernoulli or threshold.
        probs = torch.sigmoid(detached)
        if sampling == 'max_pred':
            target = (probs > 0.5).float()
        else:  # multinomial -> bernoulli sample
            target = torch.bernoulli(probs)
        return target
    if ('birads' in mod_first.lower() or 'density' in mod_first.lower() or 'diag' in mod_first.lower()):
        probs = torch.softmax(detached, dim=-1)
        if sampling == 'max_pred':
            return probs.argmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    # Binary path: out shape [bs, 2], target is class index.
    probs = torch.softmax(detached, dim=-1)
    if sampling == 'max_pred':
        return probs.argmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _compute_fisher_for_stage(model, fulltrains, encoders, criterion,
                              modalities_per_task, train_weights, args, device,
                              missing_embeddings, stage_indices, task_keys,
                              s_idx, sampling='true', num_batches=-1,
                              target_iterator=iter_expert_weights):
    """Compute the diagonal Fisher information matrix for the just-trained
    stage. Iterates the multi-task ``fulltrains`` structure, runs
    forward+backward per batch, and accumulates squared gradients per
    target parameter (whatever ``target_iterator`` yields). Returns
    ``{param_name: Fisher tensor}`` matching ``target_iterator``'s output.

    For the standard expert-only EWC (``--router_expansion`` enabled),
    pass :py:func:`iter_expert_weights`. For the no-expansion mode where
    a single shared router's weights are *also* EWC-regularized, pass
    :py:func:`iter_expert_and_router_weights`.
    """
    fisher = {n: torch.zeros_like(p, device=p.device)
              for n, p in target_iterator(model)}
    if not fulltrains:
        return fisher

    max_batches = (len(fulltrains) if num_batches is None or num_batches <= 0
                   else min(num_batches, len(fulltrains)))

    # Make sure every target weight has requires_grad so we get gradients
    # populated by the backward pass. Save and restore the prior state.
    saved_grad_state = []
    for n, p in target_iterator(model):
        saved_grad_state.append((p, p.requires_grad))
        p.requires_grad = True

    model.train()  # match training-mode behavior (dropout active, like ref).
    seen_enc_ids = set()
    for ii in stage_indices:
        enc = encoders[task_keys[ii]]
        if id(enc) not in seen_enc_ids:
            seen_enc_ids.add(id(enc))
            enc.train()

    n_samples = 0
    try:
        for batch_idx, js in enumerate(fulltrains):
            if batch_idx >= max_batches:
                break
            # Zero grads on target weights only.
            for _, p in target_iterator(model):
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

            losses = 0.0
            batch_size_total = 0
            for ii in js:
                task = task_keys[ii]
                modalities_t = modalities_per_task[ii]
                encoder_t = encoders[task]
                crit_t = criterion[ii]

                model.to_logits = model.to_logitslist[ii]
                set_current_task_idx(model, s_idx)

                embeddings, label = _encode_batch(
                    task, js[ii], encoder_t, modalities_t, device,
                )
                indict = _build_indict(
                    embeddings, modalities_t, device, args, missing_embeddings,
                )
                out, balance_loss = model(indict, task=task)

                target = _ewc_pseudo_label(
                    out, modalities_t[0], sampling, label, device,
                )
                loss = crit_t(out, target)
                if balance_loss is not None:
                    loss = loss + args.balance_loss_coef * balance_loss
                loss = loss * train_weights[ii]
                losses = losses + loss
                if hasattr(label, 'shape') and len(label.shape) > 0:
                    batch_size_total += int(label.shape[0])

            if not isinstance(losses, torch.Tensor):
                continue
            losses.backward()
            weight = max(batch_size_total, 1)
            for n, p in target_iterator(model):
                if p.grad is None:
                    continue
                fisher[n] = fisher[n] + p.grad.detach().pow(2) * weight
            n_samples += weight
    finally:
        for p, prev in saved_grad_state:
            p.requires_grad = prev

    if n_samples > 0:
        fisher = {n: f / n_samples for n, f in fisher.items()}
    return fisher


def _apply_encoder_freeze_policy(encoders, s_idx, stage_task_indices, task_keys,
                                  args, log_prefix='[CL]'):
    """Apply ``args.encoder_freeze_mode`` for stage ``s_idx`` (called *after*
    ``_freeze_all`` has zeroed every encoder param's ``requires_grad``).

    Modes:
      * ``'first_appearance'`` (default, current behavior): unfreeze encoder
        submodules for modalities whose first appearance in the stage sequence
        is the current stage; leave shared (already-trained) encoders frozen.
      * ``'all_frozen'``: keep every encoder frozen at every stage > 0. Stage
        0 still trains them because the caller skips the freeze for s_idx=0.
      * ``'all_trainable'``: unfreeze every parameter of every encoder used by
        this stage's tasks at every stage. Equivalent to letting the encoders
        co-fine-tune across all stages.
    """
    cur_indices = stage_task_indices[s_idx]
    encoder_mode = getattr(args, 'encoder_freeze_mode', 'first_appearance')

    if encoder_mode == 'all_frozen':
        print(f'{log_prefix}[stage {s_idx}] encoder_freeze_mode=all_frozen: '
              f'kept all encoders frozen.')
        return

    if encoder_mode == 'all_trainable':
        seen_enc_ids = set()
        total = 0
        thawed_keys = []
        for ii in cur_indices:
            enc = encoders[task_keys[ii]]
            eid = id(enc)
            if eid in seen_enc_ids:
                continue
            seen_enc_ids.add(eid)
            n_local = 0
            for p in enc.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    n_local += p.numel()
            total += n_local
            thawed_keys.append(task_keys[ii])
        print(f'{log_prefix}[stage {s_idx}] encoder_freeze_mode=all_trainable: '
              f'unfroze every param of encoders for {thawed_keys} '
              f'({total:,} params).')
        return

    cur_mods = set()
    for ii in cur_indices:
        cur_mods |= _task_modalities(args, task_keys[ii])
    prior_mods = set()
    for prior_s in range(s_idx):
        for ii in stage_task_indices[prior_s]:
            prior_mods |= _task_modalities(args, task_keys[ii])
    first_app = cur_mods - prior_mods

    seen_enc_ids = set()
    for ii in cur_indices:
        enc = encoders[task_keys[ii]]
        eid = id(enc)
        if eid in seen_enc_ids:
            continue
        seen_enc_ids.add(eid)
        n_un, kind, paths = _unfreeze_first_appearance_submodules(enc, first_app)
        if n_un > 0:
            print(f'{log_prefix}[stage {s_idx} -> task {task_keys[ii]}] '
                  f'unfroze encoder {kind} submodules for first-appearance '
                  f'modalities {sorted(first_app)}: {n_un:,} params, paths={paths}.')

    if not first_app:
        already = sorted(cur_mods & prior_mods)
        print(f'{log_prefix}[stage {s_idx}] kept encoders frozen '
              f'(no first-appearance modalities; already trained: {already}).')


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
    cl_method = getattr(args, 'cl_method', 'ours')

    if cl_method == 'lora':
        # LoRA baseline: unfreeze the most recently-added LoRAAdapter on
        # every LoRAExpertMLP slot. The base stays frozen (full-rank stage-0
        # pretrained expert). For stage 0 (before conversion), the slot is
        # still a plain TemporalExpertMLP -- but stage 0 doesn't reach this
        # branch (the caller only calls _set_active_for_stage for s_idx > 0).
        for sm in find_seq_moes(model):
            for slot in sm.experts:
                if isinstance(slot, LoRAExpertMLP) and len(slot.lora_adapters) > 0:
                    last_adapter = slot.lora_adapters[-1]
                    for p in last_adapter.parameters():
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
                        f"--cl_method lora requires PerTaskModalityRouter, got "
                        f"{type(r).__name__}."
                    )
    elif cl_method == 'ewc':
        # EWC baseline: experts are *always* trainable across stages; the
        # quadratic penalty (added in the training loop) keeps their
        # weights close to their post-task-(s-1) snapshot. No SVD
        # truncation, no LoRA adapter, no stacking. Slots stay as plain
        # TemporalExpertMLPs across all stages.
        router_expansion = bool(getattr(args, 'router_expansion', True))
        for sm in find_seq_moes(model):
            for slot in sm.experts:
                if isinstance(slot, TemporalExpertMLP):
                    for p in slot.parameters():
                        p.requires_grad = True
                else:
                    raise TypeError(
                        f"--cl_method ewc expects every slot to be a "
                        f"TemporalExpertMLP, got {type(slot).__name__}."
                    )
            routers = list(sm.routers.values())
            if sm.default_router is not None:
                routers.append(sm.default_router)
            for r in routers:
                if isinstance(r, PerTaskModalityRouter):
                    if router_expansion:
                        # Per-stage expansion (current behavior): unfreeze the
                        # most recently-added router head; prior heads stay
                        # frozen.
                        for p in r.task_routers[-1].parameters():
                            p.requires_grad = True
                    else:
                        # Single shared router across stages: keep
                        # ``task_routers[0]`` (the one and only) trainable
                        # so its weights move with the EWC penalty applied.
                        for p in r.task_routers[0].parameters():
                            p.requires_grad = True
                else:
                    raise TypeError(
                        f"--cl_method ewc requires PerTaskModalityRouter, got "
                        f"{type(r).__name__}."
                    )
    elif fixed_experts:
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

    _apply_encoder_freeze_policy(
        encoders, s_idx, stage_task_indices, task_keys, args, log_prefix='[CL]',
    )


def _set_active_heads_only(model, encoders, s_idx, stage_task_indices, task_keys, args):
    """Heads-only ablation: backbone (MoE experts + routers + cross-attn) stays
    frozen at random init for every stage; only the per-task classification
    heads + per-task projections of this stage's tasks train. Encoders follow
    ``args.encoder_freeze_mode`` via ``_apply_encoder_freeze_policy``.

    Caller must invoke ``_freeze_all`` first.
    """
    cur_indices = stage_task_indices[s_idx]

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

    _apply_encoder_freeze_policy(
        encoders, s_idx, stage_task_indices, task_keys, args,
        log_prefix='[CL][heads_only]',
    )


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
                                   fixed_experts=False, cl_method='ours',
                                   lora_rank=None, router_expansion=True):
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
    if cl_method == 'lora':
        # LoRA baseline: stage 0 = full-rank pretraining (no truncation),
        # stages >= 1 = additive LoRA adapters on the frozen base.
        for sm in find_seq_moes(model):
            if t_idx == 0:
                # Freeze the trained stage-0 expert and wrap as LoRA base.
                # Pool size and structure are unchanged (no SVD truncation).
                convert_to_lora_after_stage_0(sm)
            else:
                # Freeze the just-trained LoRA adapter (the last one in the
                # adapter list of every LoRAExpertMLP on this SeqMoE).
                freeze_active_lora_adapter(sm)
            freeze_router_columns(sm, n_frozen_cols=sm.num_experts, mode=mode)
        if t_idx == 0:
            print(f'[CL][cl_method=lora] froze full-rank stage-0 base on '
                  f'every slot (no SVD truncation).')
        else:
            print(f'[CL][cl_method=lora] froze stage-{t_idx} LoRA adapter on '
                  f'every slot.')

        if t_idx < num_tasks - 1:
            r = lora_rank if lora_rank is not None else rank
            for sm in find_seq_moes(model):
                append_lora_adapter(sm, r)
                add_router_head_only(sm)
            print(f'[CL][cl_method=lora] appended fresh rank-{r} LoRA adapter '
                  f'to every slot + new router head for stage {t_idx + 1}.')
        return

    if cl_method == 'ewc':
        # EWC baseline: no SVD truncation, no LoRA adapters, no stacking.
        # Expert weights stay continually trainable across stages; the EWC
        # quadratic penalty (added to the training loss in the outer loop)
        # regularizes them toward the post-task snapshot held in EWCState.
        # The Fisher snapshot + computation is done in train_continual
        # (it needs the stage's data loader). What this function does:
        #
        #   * router_expansion=True (default): freeze prior router columns
        #     and -- if not the last stage -- add a new router head for
        #     the next stage. Same as ours/lora's per-stage routing.
        #
        #   * router_expansion=False: do nothing for routers. The single
        #     shared router (task_routers[0]) stays trainable through every
        #     stage and is regularized by EWC alongside the experts.
        if router_expansion:
            for sm in find_seq_moes(model):
                freeze_router_columns(sm, n_frozen_cols=sm.num_experts, mode=mode)
            print(f'[CL][cl_method=ewc][router_expansion=True] end of stage '
                  f'{t_idx}: router columns frozen for prior stages.')
            if t_idx < num_tasks - 1:
                for sm in find_seq_moes(model):
                    add_router_head_only(sm)
                print(f'[CL][cl_method=ewc][router_expansion=True] added new '
                      f'router head for stage {t_idx + 1}; pool size unchanged '
                      f'at {next(iter(find_seq_moes(model))).num_experts}.')
        else:
            print(f'[CL][cl_method=ewc][router_expansion=False] end of stage '
                  f'{t_idx}: shared router (task_routers[0]) kept trainable; '
                  f'no new router head added.')
        return

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

    # task_keys is the FLAT per-task list in the order
    # ``setup_tasks_and_modalities`` returned (its hard-coded order:
    # ihm, los, pheno, readmission, mortality, birads, risk, density, diag).
    # task_slugs MUST be derived from task_keys (not from args.task.split('-'))
    # so that ``task_slugs[ii]`` always corresponds to ``task_keys[ii]``.
    # Otherwise prints, eval log keys, and savedir paths use slugs from the
    # user's stage order while ii indexes setup's order, producing wrong
    # labels even when the data being evaluated is correct.
    task_keys = [modalities_per_task[ii][0].split('_')[1] for ii in range(len(modalities_per_task))]
    task_slugs = [TASK_KEY_TO_SLUG.get(k, k.lower()) for k in task_keys]
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

    # EWC state: initialized once at the very start (zero Fisher, init
    # weights as the "older params"). For stages > 0 the regularizer adds a
    # quadratic penalty pulling target weights toward the post-stage-(s-1)
    # snapshot. The "target" set depends on --router_expansion:
    #   * --router_expansion (default, True): expert weights only. New router
    #     heads added per stage, so EWC has no need to constrain them
    #     (they're either freshly created or frozen from prior stages).
    #   * --no_router_expansion (False): one shared router across all stages,
    #     never expanded. EWC penalty extends to its w_gate/w_noise.
    # For non-EWC methods this stays None and is never consulted.
    ewc_state = None
    cl_method_global = getattr(args, 'cl_method', 'ours')
    router_expansion = bool(getattr(args, 'router_expansion', True))
    if cl_method_global == 'ewc' and not bool(getattr(args, 'heads_only', False)):
        target_iterator = (iter_expert_weights if router_expansion
                           else iter_expert_and_router_weights)
        ewc_state = EWCState(model, target_iterator=target_iterator)
        ewc_lamb = float(getattr(args, 'ewc_lamb', 5000.0))
        ewc_alpha = float(getattr(args, 'ewc_alpha', 0.5))
        ewc_fi_sampling = getattr(args, 'ewc_fi_sampling', 'true')
        ewc_fi_num_samples = int(getattr(args, 'ewc_fi_num_samples', -1))
        regularized_set = ('expert weights only' if router_expansion
                           else 'expert weights + shared router (w_gate, w_noise)')
        print(f'[CL][cl_method=ewc] initialized EWC state: '
              f'lamb={ewc_lamb}, alpha={ewc_alpha}, '
              f'fi_sampling={ewc_fi_sampling!r}, '
              f'fi_num_samples={ewc_fi_num_samples}, '
              f'router_expansion={router_expansion} -> regularizing {regularized_set} '
              f'({ewc_state.num_params_tracked():,} parameter values tracked).')
    else:
        ewc_lamb = 0.0
        ewc_alpha = 0.5
        ewc_fi_sampling = 'true'
        ewc_fi_num_samples = -1

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
        # ``--heads_only`` overrides this for every stage (including 0):
        # backbone stays frozen at random init, only heads + per-task
        # projections + first-appearance encoders train.
        heads_only = bool(getattr(args, 'heads_only', False))
        if heads_only:
            _freeze_all(model, encoders)
            _set_active_heads_only(
                model, encoders, s_idx, stage_task_indices, task_keys, args=args,
            )
        elif s_idx > 0:
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

                # EWC quadratic penalty on expert weights, added once per
                # batch (not per task within the batch). Skipped on stage 0
                # because Fisher is still zero and ``older_params`` equals
                # the initial expert weights -- the regularizer would be
                # zero anyway, but skipping avoids the extra forward sweep.
                if ewc_state is not None and s_idx > 0:
                    if isinstance(losses, torch.Tensor):
                        losses = losses + ewc_lamb * ewc_state.regularizer(model)
                    else:
                        losses = ewc_lamb * ewc_state.regularizer(model)

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

        # --- EWC: snapshot best params + compute/merge Fisher on the stage's
        # training data (uses the same `fulltrains` structure as training).
        # Done BEFORE the eval/reservation steps so the snapshot reflects the
        # best-on-val state, not the post-eval state. For non-EWC methods
        # this whole block is a no-op.
        if ewc_state is not None:
            # Drift diagnostic: how far did the *target* weights move from the
            # PREVIOUS stage's snapshot? Computed BEFORE snapshot_older_params
            # overwrites the anchor. The target set depends on
            # --router_expansion (expert-only vs expert+shared-router).
            with torch.no_grad():
                drift_sq = 0.0
                anchor_sq = 0.0
                max_abs_diff = 0.0
                tracked = 0
                for n, p in ewc_state.target_iterator(model):
                    if n not in ewc_state.older_params:
                        continue
                    diff = (p.detach() - ewc_state.older_params[n].to(p.device))
                    drift_sq += diff.pow(2).sum().item()
                    anchor_sq += ewc_state.older_params[n].pow(2).sum().item()
                    max_abs_diff = max(max_abs_diff, diff.abs().max().item())
                    tracked += 1
            drift_norm = drift_sq ** 0.5
            anchor_norm = anchor_sq ** 0.5
            relative = (drift_norm / anchor_norm) if anchor_norm > 0 else float('nan')
            print(f'[CL][cl_method=ewc] target-weight drift since stage '
                  f'{s_idx - 1 if s_idx > 0 else "init"}: '
                  f'||Δ||_F = {drift_norm:.4e}, max|Δ| = {max_abs_diff:.4e}, '
                  f'||Δ||/||θ_old|| = {relative:.4e} (across {tracked} weight tensors).')

            print(f'[CL][cl_method=ewc] computing Fisher on stage {s_idx} '
                  f'training data (sampling={ewc_fi_sampling!r}, '
                  f'num_batches={ewc_fi_num_samples})...')
            curr_fisher = _compute_fisher_for_stage(
                model, fulltrains, encoders, criterion, modalities_per_task,
                train_weights, args, device, missing_embeddings,
                stage_indices=stage_indices, task_keys=task_keys,
                s_idx=s_idx, sampling=ewc_fi_sampling,
                num_batches=ewc_fi_num_samples,
                target_iterator=ewc_state.target_iterator,
            )
            ewc_state.snapshot_older_params(model)
            ewc_state.merge_fisher(curr_fisher, alpha=ewc_alpha)
            mean_fisher = (
                sum(f.mean().item() for f in ewc_state.fisher.values())
                / max(len(ewc_state.fisher), 1)
            )
            max_fisher = (
                max((f.max().item() for f in ewc_state.fisher.values()),
                    default=0.0)
            )
            print(f'[CL][cl_method=ewc] snapshot taken; Fisher merged with '
                  f'alpha={ewc_alpha} (running Fisher mean = {mean_fisher:.3e}, '
                  f'max = {max_fisher:.3e}).')

            # Also log the EWC penalty value the *next* stage would see at
            # its first batch (i.e., evaluating the regularizer at the post-
            # snapshot state, with merged Fisher). This should be ~0 since
            # we just snapshotted; non-zero values indicate a bug.
            with torch.no_grad():
                check_reg = ewc_state.regularizer(model).item()
            print(f'[CL][cl_method=ewc] regularizer at snapshot point '
                  f'(should be ~0): {check_reg:.4e}')

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

        # --- (2) Post-stage: SVD-reserve, freeze, grow (or LoRA / EWC equivalents).
        # Skipped under --heads_only: there is no backbone training to
        # snapshot/reserve, so the model stays at its random-init structure
        # for every stage.
        if not bool(getattr(args, 'heads_only', False)):
            _post_task_reserve_freeze_grow(
                model, s_idx, num_tasks=num_stages,
                num_experts_per_task=num_experts_per_stage, rank=rank, mode=mode,
                fixed_experts=bool(getattr(args, 'fixed_experts', False)),
                cl_method=getattr(args, 'cl_method', 'ours'),
                lora_rank=getattr(args, 'lora_cl_rank', None),
                router_expansion=bool(getattr(args, 'router_expansion', True)),
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
        # Mirror the multi-task ``best_model_results_*.txt`` output format
        # under args.results_dir so aggregate_results.py-style tooling can
        # scan continual runs. Run AFTER both full-rank and reserved-rank
        # evals have populated cl_log so the writer can pull from whichever
        # phase the cl_method dictates (reserved for ours, full_rank for lora).
        _write_best_model_results_for_stage(
            args, s_idx, stage_label,
            stage_task_indices=stage_task_indices,
            task_keys=task_keys, task_slugs=task_slugs,
            cl_log=cl_log,
        )

    _print_and_write_progression_table(
        args, savedir_root, num_stages,
        stage_task_indices=stage_task_indices,
        task_keys=task_keys, task_slugs=task_slugs,
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
