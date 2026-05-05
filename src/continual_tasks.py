"""Continual learning entry point for clinical-highmmt.

Sibling of ``mimiciv_tasks.py``: builds the same multitask model
(``MULTCrossModel`` with ``--multitask_moe`` enabled, i.e., the SeqMoE
backbone) and then trains tasks *sequentially* in the order given by
``--task`` (e.g., ``ihm-los-birads``). Step-1 ships only the orchestrator;
the rank-reservation, expert-pool growth, and per-task routing-mask machinery
are added in later steps and plumbed through the same args defined here.

Existing pipelines (``run_flame_embed.sh`` -> ``mimiciv_tasks.py``) are not
modified -- this is a parallel entry point.
"""

import argparse
import datetime
import os
import sys

sys.path.insert(1, os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from accelerate import Accelerator
from transformers import set_seed

from src.continual.cl_moe import install_continual_routers
from src.continual.cl_train import train_continual
from src.fusemoe import MULTCrossModel
from src.mimiciv_task_setup import setup_tasks_and_modalities
from src.mimiciv_tasks import loadBert, parse_args as base_parse_args


def _task_to_mod_arg(t):
    return {'readmission': 'rad_mod', 'mortality': 'mor_mod'}.get(t, f'{t}_mod')


# Stage-parsing helpers live in src/continual/cl_stages.py so they can be
# unit-tested without pulling in the heavy torch/mimiciv_tasks transitive
# imports. Re-exported here for backwards compatibility with any code that
# imports them from continual_tasks.
from src.continual.cl_stages import (  # noqa: E402
    SLUG_TO_TASK_KEY,
    parse_task_sequence,
    flatten_task_arg,
    task_sequence_to_stage_indices,
    path_safe_task_str as _path_safe_task_str,
)


class _CLLogTee:
    """Wrap a stdout stream and mirror lines starting with '[CL]' to a log file.

    Other output passes through to the original stream untouched. We buffer
    partial writes until we see a newline because `print` may flush in
    multiple chunks (and some [CL] messages are preceded by '\\n' separators).
    """

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self._buffer = ''

    def write(self, s):
        self.stream.write(s)
        self._buffer += s
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            if line.lstrip().startswith('[CL]'):
                self.log_file.write(line + '\n')
                self.log_file.flush()
        return len(s) if isinstance(s, str) else 0

    def flush(self):
        self.stream.flush()
        try:
            self.log_file.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _install_cl_logger(args, modeltype_str):
    """Open a log file mirroring the checkpoint savedir convention and tee
    `[CL]`-prefixed stdout lines to it.

    The path mirrors `savedir_root` in `main()`: same component layout, rooted
    at <src>/logs/ instead of ./checkpoints/, leaf is a `.log` file rather
    than a directory. Appends so re-running the same config preserves history.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(
        src_dir, 'logs', 'continual',
        args.gating_function[0] if args.gating_function else 'default',
        _path_safe_task_str(getattr(args, 'task_raw', args.task)), modeltype_str,
    )
    os.makedirs(log_dir, exist_ok=True)
    cl_method = getattr(args, 'cl_method', 'ours')
    enc_mode = getattr(args, 'encoder_freeze_mode', 'first_appearance')
    enc_tag = '' if enc_mode == 'first_appearance' else f'_enc_{enc_mode}'
    cl_target = getattr(args, 'cl_target', getattr(args, 'svd_target', 'moe'))
    target_tag = '' if cl_target == 'moe' else f'_target_{cl_target}'
    cl_method_tag = (
        f'cl_{cl_method}'
        + (f'_lorarank{args.lora_cl_rank}' if cl_method == 'lora' else '')
        + (f'_lamb{args.ewc_lamb}_alpha{args.ewc_alpha}_fi_{args.ewc_fi_sampling}'
           if cl_method == 'ewc' else '')
        + ('_fixed_experts' if getattr(args, 'fixed_experts', False) else '')
        + ('_heads_only' if getattr(args, 'heads_only', False) else '')
        + enc_tag
        + target_tag
    )
    log_filename = (
        f'{cl_method_tag}_router_{args.router_growth_mode}_rank{args.reserved_rank}_'
        f'replay{args.replay_proportion}_alpha{args.alpha}_'
        f'lr{args.lr}_wd{args.weight_decay}_'
        f'mod_drop_rate_{args.modality_drop_rate}.log'
    )
    log_path = os.path.join(log_dir, log_filename)
    log_file = open(log_path, 'a', buffering=1)
    log_file.write(f'\n===== Session start: {datetime.datetime.now().isoformat()} =====\n')
    log_file.flush()
    sys.stdout = _CLLogTee(sys.stdout, log_file)
    return log_path


def parse_cl_args():
    """Reuse mimiciv_tasks.parse_args; layer continual-specific args on top."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        '--reserved_rank', type=int, default=16,
        help='Uniform SVD rank used to compress trained MoE experts after each task. '
             'Used by Step-2+ rank reservation (no-op in Step-1).',
    )
    pre.add_argument(
        '--router_growth_mode',
        choices=['column_grow', 'per_task_router'], default='column_grow',
        help='How to grow modality routers across tasks. column_grow: append columns to a '
             'shared w_gate/w_noise. per_task_router: separate ModalityRouter per task.',
    )
    pre.add_argument(
        '--replay_proportion', type=float, default=0.0,
        help='Fraction of previous-task samples to replay during current task. '
             '0.0 = no replay (current default).',
    )
    pre.add_argument(
        '--router_combine',
        choices=['mean', 'sum', 'max'], default='mean',
        help='Combine operator over per-task router logits at inference '
             '(per_task_router mode only). mean: per-expert mean across '
             'routers that see the expert. sum: per-expert sum. max: per-expert max.',
    )
    pre.add_argument(
        '--fixed_experts', action='store_true',
        help='Keep the expert pool size fixed at --num_of_experts across all '
             'tasks. Each slot becomes a StackedExpertMLP whose components are '
             'one rank-r reserved expert per completed task. New tasks add a '
             'fresh, zero-initialised TemporalExpertMLP component (whose output '
             'is added to the prior components) instead of growing the pool. '
             'Requires --router_growth_mode per_task_router.',
    )
    pre.add_argument(
        '--cl_method',
        choices=['ours', 'lora', 'ewc'], default='ours',
        help='Continual-learning method to compare. ours: per-stage components '
             'with stacked low-rank reservation (the default pipeline). lora: '
             'per-stage additive LoRA adapters on the stage-0 (full-rank) base '
             'expert -- biases/LayerNorm shared with the base, only the three '
             'expert weight matrices (temporal_conv, fc1, fc2) get LoRA. ewc: '
             'Elastic Weight Consolidation -- experts continually trainable '
             'across stages with a quadratic penalty on the same three weight '
             'matrices toward their post-stage snapshot, weighted by the '
             'diagonal Fisher information matrix. Requires --fixed_experts.',
    )
    pre.add_argument(
        '--lora_cl_rank', type=int, default=None,
        help='Rank for the per-stage LoRA adapters when --cl_method=lora. '
             'Defaults to --reserved_rank if unspecified.',
    )
    pre.add_argument(
        '--ewc_lamb', type=float, default=5000.0,
        help='EWC regularization strength lambda when --cl_method=ewc '
             '(default matches reference, arXiv:1612.00796).',
    )
    pre.add_argument(
        '--ewc_alpha', type=float, default=0.5,
        help='EWC Fisher fusion weight when --cl_method=ewc. '
             'Fisher_new = alpha * Fisher_old + (1 - alpha) * Fisher_curr. '
             '0.5 = equal weight (default), 1.0 = freeze old Fisher, '
             '0.0 = forget old Fisher.',
    )
    pre.add_argument(
        '--ewc_fi_sampling',
        choices=['true', 'max_pred', 'multinomial'], default='true',
        help='Sampling type for Fisher information when --cl_method=ewc. '
             'true: use ground-truth labels with task criterion (default). '
             'max_pred: use model argmax/threshold prediction. '
             'multinomial: sample from softmax/sigmoid output distribution.',
    )
    pre.add_argument(
        '--ewc_fi_num_samples', type=int, default=-1,
        help='Number of training batches to use when computing Fisher '
             '(-1 = all batches in the stage). Smaller values trade Fisher '
             'estimate quality for compute.',
    )
    pre.add_argument(
        '--router_expansion', action=argparse.BooleanOptionalAction, default=True,
        help='Whether each stage adds a new ModalityRouter head '
             '(per_task_router) or reuses a single shared router. Default '
             '(True) matches all three CL methods\' current behavior. With '
             '--no-router_expansion (only meaningful for --cl_method=ewc), '
             'the same task_routers[0] is continually fine-tuned across all '
             'stages and EWC regularizes its w_gate / w_noise alongside the '
             'expert weights.',
    )
    pre.add_argument(
        '--heads_only', action='store_true',
        help='Heads-only ablation: freeze the entire MoE + cross-attn '
             'backbone at random init for every stage (including stage 0). '
             'Only the per-task classification heads + per-task projections '
             'train. Encoders follow --encoder_freeze_mode. Disables EWC / '
             'LoRA / SVD reservation regardless of --cl_method, since there '
             'is no backbone training to regularize or reserve.',
    )
    pre.add_argument(
        '--cl_target', '--svd_target',
        dest='cl_target',
        choices=['moe', 'moe_and_encoder'], default='moe',
        help='Where the cl_method-specific machinery runs at end of each '
             'stage. moe (default): only MoE expert weights are protected '
             '(SVD-stacked for ours / LoRA-adapted for lora / Fisher-'
             'regularized for ewc). moe_and_encoder: also extends the same '
             'protection to encoder nn.Linear and nn.Conv1d(kernel_size=1) '
             'layers that were trainable this stage, in the cl_method\'s '
             'native style (StackedLowRank* for ours; per-stage LoRA '
             'adapters for lora; Fisher penalty extended to encoder weights '
             'for ewc). Overrides --encoder_freeze_mode for layers it '
             'touches. ``--svd_target`` is kept as a deprecated alias.',
    )
    pre.add_argument(
        '--encoder_freeze_mode',
        choices=['first_appearance', 'all_frozen', 'all_trainable'],
        default='first_appearance',
        help='How modality encoders are handled at stage > 0 (and at every '
             'stage under --heads_only). first_appearance (default): unfreeze '
             'only encoder submodules whose first appearance in the stage '
             'sequence is the current stage; encoders shared with prior '
             'stages stay frozen. all_frozen: keep all encoders frozen at '
             'every stage > 0 (only stage 0 trains them in the standard path; '
             'with --heads_only, stage 0 also keeps encoders frozen). '
             'all_trainable: unfreeze every parameter of every encoder used '
             'by this stage at every stage -- full encoder fine-tuning '
             'across stages.',
    )
    cl_only, remaining = pre.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = base_parse_args()
    args.reserved_rank = cl_only.reserved_rank
    args.router_growth_mode = cl_only.router_growth_mode
    args.replay_proportion = cl_only.replay_proportion
    args.router_combine = cl_only.router_combine
    args.fixed_experts = cl_only.fixed_experts
    args.cl_method = cl_only.cl_method
    args.lora_cl_rank = (
        cl_only.lora_cl_rank if cl_only.lora_cl_rank is not None
        else cl_only.reserved_rank
    )
    args.ewc_lamb = cl_only.ewc_lamb
    args.ewc_alpha = cl_only.ewc_alpha
    args.ewc_fi_sampling = cl_only.ewc_fi_sampling
    args.ewc_fi_num_samples = cl_only.ewc_fi_num_samples
    args.router_expansion = cl_only.router_expansion
    args.heads_only = cl_only.heads_only
    args.encoder_freeze_mode = cl_only.encoder_freeze_mode
    args.cl_target = cl_only.cl_target
    args.svd_target = cl_only.cl_target  # deprecated alias preserved
    if not args.router_expansion and args.cl_method != 'ewc':
        # For ours/lora, per-stage router expansion is structural -- they
        # rely on per_task_router heads to isolate prior tasks. Disabling
        # it would break their continual semantics. Quietly re-enable and
        # warn the user.
        print(f'[CL] warn: --no-router_expansion is only meaningful for '
              f'--cl_method=ewc; ignoring for cl_method={args.cl_method!r} '
              '(re-enabling per-stage router expansion).')
        args.router_expansion = True
    if args.fixed_experts and args.router_growth_mode != 'per_task_router':
        raise ValueError(
            '--fixed_experts requires --router_growth_mode per_task_router. '
            "column_grow's gradient-mask freezing is tied to a growing w_gate "
            'and has nothing to grow when the pool is fixed.'
        )
    if args.cl_method == 'lora' and not args.fixed_experts:
        raise ValueError(
            '--cl_method lora requires --fixed_experts (LoRA assumes a fixed '
            'pool of E experts that LoRA adapters attach to; pool growth is '
            'incompatible with the LoRA-on-base baseline).'
        )
    if args.cl_method == 'lora' and args.router_growth_mode != 'per_task_router':
        raise ValueError(
            '--cl_method lora requires --router_growth_mode per_task_router '
            '(matches the per-stage router-head structure used by --fixed_experts).'
        )
    if args.cl_method == 'ewc' and not args.fixed_experts:
        raise ValueError(
            '--cl_method ewc requires --fixed_experts (EWC continually fine-tunes '
            'a fixed pool of E experts under a quadratic penalty; pool growth '
            'is incompatible with this regularization paradigm).'
        )
    if args.cl_method == 'ewc' and args.router_growth_mode != 'per_task_router':
        raise ValueError(
            '--cl_method ewc requires --router_growth_mode per_task_router '
            '(matches the per-stage router-head structure used by --fixed_experts).'
        )
    args.continual = True

    # Preserve the user-facing task string (with ``;`` for stage boundaries)
    # for logging, and derive the flat hyphen-only form that
    # ``setup_tasks_and_modalities`` expects on its existing args.task path.
    args.task_raw = args.task
    args.task_stages = parse_task_sequence(args.task_raw)
    args.task_stage_indices = task_sequence_to_stage_indices(args.task_stages)
    # ``args.task`` is overwritten with the flat form so downstream code that
    # was already wired for the existing setup keeps working.
    args.task = flatten_task_arg(args.task_raw)
    return args


def _build_modeltype_and_modalities(args):
    """Build the ``modeltype`` dict and the union ``modalities`` set for the
    current task list, mirroring the logic at the top of ``mimiciv_tasks.main``.
    """
    modalities = set()
    modeltype = {}
    pairs = [
        ('ihm', 'ihm_mod'), ('los', 'los_mod'), ('pheno', 'pheno_mod'),
        ('readmission', 'rad_mod'), ('mortality', 'mor_mod'),
        ('birads', 'birads_mod'), ('risk', 'risk_mod'), ('density', 'density_mod'),
    ]
    requested = set(args.task.split('-'))
    for t, attr in pairs:
        v = getattr(args, attr, '') or ''
        if v and t in requested:
            modeltype[t] = '_'.join(sorted(v.split('-')))
            for e in v.split('-'):
                modalities.add(e)
    return modeltype, modalities


def main():
    args = parse_cl_args()
    set_seed(args.seed)
    args.mixed_precision = 'fp16' if args.fp16 else 'no'
    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=args.cpu)
    device = accelerator.device

    # Pre-flight checks: continual pipeline relies on shared encoders + multitask MoE.
    if not args.shared_modality_encoders:
        raise ValueError('Continual pipeline requires --shared_modality_encoders.')
    if args.fusion_model != 'fusemoe':
        raise ValueError("Continual pipeline currently supports --fusion_model 'fusemoe' only.")
    if not args.multitask_moe:
        raise ValueError('Continual pipeline requires --multitask_moe (SeqMoE backbone).')

    modeltype, modalities = _build_modeltype_and_modalities(args)

    if 'Text' in modalities:
        BioBert, _, tokenizer = loadBert(args, device)
    else:
        BioBert, tokenizer = None, None

    (
        all_train, all_valid, all_test, criterion,
        modalities_per_task, train_weights, all_encoders, logits, all_modalities,
    ) = setup_tasks_and_modalities(
        args=args, device=device, tokenizer=tokenizer,
        modeltype=modeltype, modalities=modalities, BioBert=BioBert,
    )

    # Re-align stage indices to the *actual* flat order returned by
    # setup_tasks_and_modalities. Two bugs we're correcting here:
    #
    #   (1) setup recognizes a fixed vocabulary of slugs in its own order
    #       (ihm, los, pheno, readmission, mortality, birads, risk, density,
    #       diag). Slugs not in this set silently produce no per-task entry,
    #       so e.g. --task 'ihm-rad-birads' loads only IHM + BIRADS (the
    #       'rad' slug is not 'readmission' and is silently ignored). We
    #       error early instead of letting an IndexError surface later.
    #
    #   (2) setup returns the per-task arrays in *its* fixed order, NOT in
    #       the user's stage order. e.g. --task 'los;birads;ihm' parses to
    #       stages [['los'], ['birads'], ['ihm']] but setup returns
    #       per-task arrays ordered [IHM, LOS, BIRADS]. The pre-setup blind
    #       indexing in parse_cl_args mapped stage 0 -> flat idx 0 (IHM),
    #       which is wrong: the user expected stage 0 to mean 'los'.
    #
    # Fix: build a slug -> flat-index map from the actual task_keys (uppercase
    # short names that setup uses) and remap args.task_stage_indices.
    flat_task_keys = [
        modalities_per_task[ii][0].split('_')[1]
        for ii in range(len(modalities_per_task))
    ]
    key_to_flat_idx = {key: idx for idx, key in enumerate(flat_task_keys)}
    remapped_indices = []
    bad = []
    for stage in args.task_stages:
        stage_flat = []
        for slug in stage:
            slug_norm = slug.lower()
            key = SLUG_TO_TASK_KEY.get(slug_norm)
            if key is None or key not in key_to_flat_idx:
                bad.append(slug)
                continue
            stage_flat.append(key_to_flat_idx[key])
        remapped_indices.append(stage_flat)
    if bad:
        recognized = sorted(SLUG_TO_TASK_KEY.keys())
        raise ValueError(
            f'Unrecognized task slug(s) in --task: {bad}. '
            f'Recognized slugs are: {recognized}. '
            f'Note: use "readmission" (not "rad") for the readmission task '
            f'and "mortality" (not "mor") for mortality.'
        )
    if any(len(s) == 0 for s in remapped_indices):
        raise ValueError(
            f'After remapping, some stages contain zero tasks: {remapped_indices}. '
            f'Original stages: {args.task_stages}.'
        )
    args.task_stage_indices = remapped_indices
    print(f'[CL] aligned stage indices to setup\'s flat order: '
          f'flat task_keys={flat_task_keys}, '
          f'stage_task_indices={args.task_stage_indices}.')

    # Replicate the modeltype string the multitask path passes to MULTCrossModel:
    # concatenate each task's modality string with '_' separators.
    mod_strs = [getattr(args, _task_to_mod_arg(t)) for t in args.task.split('-')]
    modeltype_str = '_'.join(mod_strs)

    # Tee [CL]-prefixed stdout to a log file before any [CL] prints fire below.
    log_path = _install_cl_logger(args, modeltype_str)
    print(f'[CL] Logging [CL]-prefixed output to: {log_path}')

    # ``perceiver_mod`` mirrors the shared-encoder branch in mimiciv_tasks.main:
    # union of modality names across all tasks (sorted for deterministic order).
    shared_modalities = sorted(set(
        m for tm in modeltype_str.split('_') for m in tm.split('-')
    ))
    perceiver_mod = [all_modalities[m] for m in shared_modalities]

    model = MULTCrossModel(
        args, device,
        modeltype=modeltype_str,
        modalities=perceiver_mod,
        modalities_per_task=modalities_per_task,
        num_classes=1,
    ).to(device)
    model.to_logitslist = logits.to(device)

    # Replace every vanilla ModalityRouter inside every SeqMoE block with the
    # selected continual variant before training begins. Migrates state, so
    # forward at task 0 is bit-identical to the vanilla pipeline.
    install_continual_routers(
        model, mode=args.router_growth_mode, combine=args.router_combine,
    )
    print(f'[CL] Installed {args.router_growth_mode!r} continual routers '
          f'(combine={args.router_combine!r} for per_task_router) on all SeqMoE blocks.')

    enc_mode = getattr(args, 'encoder_freeze_mode', 'first_appearance')
    enc_tag = '' if enc_mode == 'first_appearance' else f'_enc_{enc_mode}'
    cl_target = getattr(args, 'cl_target', getattr(args, 'svd_target', 'moe'))
    target_tag = '' if cl_target == 'moe' else f'_target_{cl_target}'
    cl_method_tag = (
        f'cl_{args.cl_method}'
        + (f'_lorarank{args.lora_cl_rank}' if args.cl_method == 'lora' else '')
        + (f'_lamb{args.ewc_lamb}_alpha{args.ewc_alpha}_fi_{args.ewc_fi_sampling}'
           f'{"_no_router_exp" if not args.router_expansion else ""}'
           if args.cl_method == 'ewc' else '')
        + ('_heads_only' if getattr(args, 'heads_only', False) else '')
        + enc_tag
        + target_tag
    )
    savedir_root = (
        f'./checkpoints/continual/{args.gating_function[0] if args.gating_function else "default"}/'
        f'{_path_safe_task_str(getattr(args, "task_raw", args.task))}/{modeltype_str}/{args.seed}/'
        f'{cl_method_tag}_router_{args.router_growth_mode}'
        f'{"_fixed_experts" if args.fixed_experts else ""}'
        f'_rank{args.reserved_rank}_'
        f'replay{args.replay_proportion}_alpha{args.alpha}_'
        f'lr{args.lr}_wd{args.weight_decay}_'
        f'mod_drop_rate_{args.modality_drop_rate}'
    )
    os.makedirs(savedir_root, exist_ok=True)
    print(f'[CL] Checkpoints will be written under: {savedir_root}')

    train_continual(
        model=model,
        all_train=all_train, all_valid=all_valid, all_test=all_test,
        modalities_per_task=modalities_per_task,
        criterion=criterion,
        train_weights=train_weights,
        encoders=all_encoders,
        args=args,
        savedir_root=savedir_root,
        device=device,
    )
    print('[CL] All tasks complete.')


if __name__ == '__main__':
    main()
