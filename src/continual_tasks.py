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
    log_filename = (
        f'router_{args.router_growth_mode}_rank{args.reserved_rank}_'
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
    cl_only, remaining = pre.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = base_parse_args()
    args.reserved_rank = cl_only.reserved_rank
    args.router_growth_mode = cl_only.router_growth_mode
    args.replay_proportion = cl_only.replay_proportion
    args.router_combine = cl_only.router_combine
    args.fixed_experts = cl_only.fixed_experts
    if args.fixed_experts and args.router_growth_mode != 'per_task_router':
        raise ValueError(
            '--fixed_experts requires --router_growth_mode per_task_router. '
            "column_grow's gradient-mask freezing is tied to a growing w_gate "
            'and has nothing to grow when the pool is fixed.'
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

    savedir_root = (
        f'./checkpoints/continual/{args.gating_function[0] if args.gating_function else "default"}/'
        f'{_path_safe_task_str(getattr(args, "task_raw", args.task))}/{modeltype_str}/'
        f'router_{args.router_growth_mode}'
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
