"""Unit tests for the multi-task-stage parsing helpers in
``continual_tasks``. These don't import ``mimiciv_tasks`` (which has the
heavy GLIBC-tied flexmoe transitive dep), so they run in any env.

Run from project root with:
    python -m src.continual.test_stage_parsing
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the parsing helpers from the lightweight module. Avoids the heavy
# torch/mimiciv_tasks transitive imports that ``continual_tasks`` triggers.
from src.continual.cl_stages import (
    parse_task_sequence,
    flatten_task_arg,
    task_sequence_to_stage_indices,
    path_safe_task_str as _path_safe_task_str,
)


def _eq(actual, expected, label):
    assert actual == expected, f'{label}: expected {expected}, got {actual}'
    print(f'  {label}: OK ({actual})')


def test_parse_task_sequence_backward_compat():
    """Without ``;``, every ``-``-separated task is its own single-task stage,
    matching the prior pipeline exactly."""
    print('parse_task_sequence backward compat (no ";")')
    print('-' * 60)
    _eq(parse_task_sequence('ihm'), [['ihm']], 'single task')
    _eq(parse_task_sequence('ihm-los'),
        [['ihm'], ['los']], '2 tasks, no ";"')
    _eq(parse_task_sequence('ihm-los-birads'),
        [['ihm'], ['los'], ['birads']], '3 tasks, no ";" (current default)')


def test_parse_task_sequence_with_stages():
    """``;`` introduces stages; ``-`` within a stage means joint multi-task."""
    print('\nparse_task_sequence with ";" stages')
    print('-' * 60)
    _eq(parse_task_sequence('ihm-los;birads'),
        [['ihm', 'los'], ['birads']], 'stage 0 multi-task, stage 1 single')
    _eq(parse_task_sequence('ihm;los-birads'),
        [['ihm'], ['los', 'birads']], 'stage 0 single, stage 1 multi-task')
    _eq(parse_task_sequence('ihm-los;birads-risk'),
        [['ihm', 'los'], ['birads', 'risk']], 'both stages multi-task')
    _eq(parse_task_sequence('ihm;los;birads'),
        [['ihm'], ['los'], ['birads']], '3 explicit single-task stages')
    _eq(parse_task_sequence('ihm-los-birads;density'),
        [['ihm', 'los', 'birads'], ['density']], 'stage 0 has 3 joint tasks')


def test_flatten_task_arg():
    """``flatten_task_arg`` produces the hyphen-only form passed to the
    existing ``setup_tasks_and_modalities``."""
    print('\nflatten_task_arg')
    print('-' * 60)
    _eq(flatten_task_arg('ihm-los-birads'), 'ihm-los-birads', 'no-op without ";"')
    _eq(flatten_task_arg('ihm-los;birads'), 'ihm-los-birads', 'replaces ";"')
    _eq(flatten_task_arg('ihm;los;birads'), 'ihm-los-birads', 'multiple ";"')
    _eq(flatten_task_arg('ihm-los;birads-risk'), 'ihm-los-birads-risk',
        'mixed stages')


def test_task_sequence_to_stage_indices():
    """Flat task indices grouped by stage."""
    print('\ntask_sequence_to_stage_indices')
    print('-' * 60)
    _eq(task_sequence_to_stage_indices([['ihm']]),
        [[0]], '1 task 1 stage')
    _eq(task_sequence_to_stage_indices([['ihm'], ['los'], ['birads']]),
        [[0], [1], [2]], '3 single-task stages')
    _eq(task_sequence_to_stage_indices([['ihm', 'los'], ['birads']]),
        [[0, 1], [2]], 'stage 0 has 2 tasks')
    _eq(task_sequence_to_stage_indices([['ihm'], ['los', 'birads']]),
        [[0], [1, 2]], 'stage 1 has 2 tasks')
    _eq(task_sequence_to_stage_indices([['ihm', 'los'], ['birads', 'risk']]),
        [[0, 1], [2, 3]], 'both stages multi-task')


def test_path_safe_task_str():
    """``;`` -> ``__`` for filesystem safety; everything else passes through."""
    print('\n_path_safe_task_str')
    print('-' * 60)
    _eq(_path_safe_task_str('ihm-los-birads'),
        'ihm-los-birads', 'no-op without ";"')
    _eq(_path_safe_task_str('ihm-los;birads'),
        'ihm-los__birads', '";" -> "__"')
    _eq(_path_safe_task_str('ihm;los;birads'),
        'ihm__los__birads', 'multiple')


def test_full_pipeline_alignment():
    """End-to-end check that for ``--task 'ihm-los;birads'``, parse +
    flatten + indices give consistent task ordering: the flat task
    index ``ii`` corresponds to slot ``ii`` on every per-task array
    returned by ``setup_tasks_and_modalities``.
    """
    print('\nfull pipeline alignment check')
    print('-' * 60)
    raw = 'ihm-los;birads'
    stages = parse_task_sequence(raw)
    flat = flatten_task_arg(raw)
    indices = task_sequence_to_stage_indices(stages)

    # The flat string is what setup_tasks_and_modalities sees.
    flat_tasks = flat.split('-')
    assert flat_tasks == ['ihm', 'los', 'birads'], flat_tasks
    # Stage indices must point to the correct tasks in the flat array.
    for s_idx, stage in enumerate(stages):
        for local_pos, expected_slug in enumerate(stage):
            global_idx = indices[s_idx][local_pos]
            assert flat_tasks[global_idx] == expected_slug, (
                f'stage {s_idx} pos {local_pos}: expected {expected_slug}, '
                f'flat[{global_idx}]={flat_tasks[global_idx]}'
            )
    print('  global flat indices line up with stage slug assignments  [OK]')


def main():
    print('Multi-task-stage parsing tests')
    print('=' * 60)
    test_parse_task_sequence_backward_compat()
    test_parse_task_sequence_with_stages()
    test_flatten_task_arg()
    test_task_sequence_to_stage_indices()
    test_path_safe_task_str()
    test_full_pipeline_alignment()
    print('\nAll stage parsing tests passed.')


if __name__ == '__main__':
    main()
