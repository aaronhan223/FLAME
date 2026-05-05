"""Smoke test that the CL ``best_model_results_*.txt`` writer produces files
``aggregate_results.parse_best_block`` can parse back to the right per-task
metric dicts.

Run from the project root with:
    python -m src.continual.test_best_model_results
"""

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.continual.cl_train import _write_best_model_results_for_stage
from src.aggregate_results import parse_best_block


def _fake_args(tmp_results_dir):
    a = types.SimpleNamespace()
    a.task = 'ihm-los-birads'
    a.task_raw = 'ihm-los;birads'
    a.cl_method = 'lora'
    a.lora_cl_rank = 16
    a.fusion_model = 'fusemoe'
    a.gating_function = ['laplace']
    a.seed = 42
    a.num_of_experts = [5]
    a.modality_drop_rate = 0.0
    a.lr = 0.0001
    a.weight_decay = 0.1
    a.reserved_rank = 16
    a.router_growth_mode = 'per_task_router'
    a.fixed_experts = True
    a.replay_proportion = 0.0
    a.alpha = 'const_0.0'
    a.results_dir = tmp_results_dir
    a.ihm_mod = 'TS-Text'
    a.los_mod = 'TS-CXR'
    a.birads_mod = 'cc-mlo-2dcc-2dmlo'
    return a


def test_writer_output_parses_back():
    with tempfile.TemporaryDirectory() as tmp:
        args = _fake_args(tmp)
        # Stage 1: tasks ihm (idx 0) and los (idx 1) trained jointly in stage
        # 0; birads (idx 2) trained in stage 1. We're writing the file at
        # the end of stage 0.
        stage_task_indices = [[0, 1], [2]]
        task_keys = ['IHM', 'LOS', 'BIRADS']
        task_slugs = ['ihm', 'los', 'birads']
        # cl_method='lora' pulls from full_rank/ entries.
        cl_log = {
            'after_stage_0/full_rank/test/ihm/primary': 0.7981,
            'after_stage_0/full_rank/test/ihm/auc': 0.7981,
            'after_stage_0/full_rank/test/ihm/auprc': 0.4513,
            'after_stage_0/full_rank/test/ihm/f1': 0.2727,
            'after_stage_0/full_rank/test/ihm/accuracy': 0.8559,
            'after_stage_0/full_rank/test/los/primary': 0.7630,
            'after_stage_0/full_rank/test/los/auc': 0.7630,
            'after_stage_0/full_rank/test/los/auprc': 0.6285,
            'after_stage_0/full_rank/test/los/f1': 0.0970,
            'after_stage_0/full_rank/test/los/accuracy': 0.5689,
        }
        out_fname = _write_best_model_results_for_stage(
            args, s_idx=0, stage_label='ihm-los',
            stage_task_indices=stage_task_indices,
            task_keys=task_keys, task_slugs=task_slugs,
            cl_log=cl_log,
        )
        assert os.path.exists(out_fname), f'file not written: {out_fname}'
        # Path layout check: stage above seed.
        assert '/stage0_ihm-los/42/' in out_fname, out_fname
        # Aggregator-compatible parse back.
        parsed = parse_best_block(out_fname)
        assert parsed is not None, (
            'aggregate_results.parse_best_block could not find a "Final Best '
            'Model Test" block in our output.'
        )
        # Two tasks, IHM (0) and LOS (1).
        assert set(parsed.keys()) == {0, 1}, parsed.keys()
        for tidx, expected_auc in [(0, 0.7981), (1, 0.7630)]:
            assert 'auc' in parsed[tidx], parsed[tidx]
            assert abs(parsed[tidx]['auc'] - expected_auc) < 1e-4
            assert 'auprc' in parsed[tidx]
            assert 'f1' in parsed[tidx]
            assert 'accuracy' in parsed[tidx]
        print(f'  wrote: {out_fname}')
        print(f'  parsed back: tasks={list(parsed.keys())}, '
              f'metrics_per_task={ {t: sorted(parsed[t].keys()) for t in parsed} }')
        print('  [OK]')


def test_two_stages_separate_paths():
    """Writing for stage 0 and then stage 1 must produce files in two
    different ``stage{s}_<label>/<seed>/`` directories."""
    with tempfile.TemporaryDirectory() as tmp:
        args = _fake_args(tmp)
        stage_task_indices = [[0, 1], [2]]
        task_keys = ['IHM', 'LOS', 'BIRADS']
        task_slugs = ['ihm', 'los', 'birads']
        cl_log = {}
        # Stage 0
        for slug, primary in [('ihm', 0.79), ('los', 0.76)]:
            cl_log[f'after_stage_0/full_rank/test/{slug}/primary'] = primary
            cl_log[f'after_stage_0/full_rank/test/{slug}/auc'] = primary
        f0 = _write_best_model_results_for_stage(
            args, s_idx=0, stage_label='ihm-los',
            stage_task_indices=stage_task_indices,
            task_keys=task_keys, task_slugs=task_slugs, cl_log=cl_log,
        )
        # Stage 1: includes all 3 tasks now (full-rank eval covers prior stages).
        for slug, primary in [('ihm', 0.78), ('los', 0.75), ('birads', 0.77)]:
            cl_log[f'after_stage_1/full_rank/test/{slug}/primary'] = primary
            cl_log[f'after_stage_1/full_rank/test/{slug}/auc'] = primary
        f1 = _write_best_model_results_for_stage(
            args, s_idx=1, stage_label='birads',
            stage_task_indices=stage_task_indices,
            task_keys=task_keys, task_slugs=task_slugs, cl_log=cl_log,
        )
        assert f0 != f1
        assert '/stage0_ihm-los/42/' in f0
        assert '/stage1_birads/42/' in f1
        # Stage 1 file should contain 3 tasks (ihm, los, birads)
        parsed = parse_best_block(f1)
        assert set(parsed.keys()) == {0, 1, 2}, parsed.keys()
        print(f'  stage 0 file: {f0}')
        print(f'  stage 1 file: {f1}')
        print(f'  stage 1 parsed tasks: {sorted(parsed.keys())}')
        print('  [OK]')


def test_ours_pulls_reserved_phase():
    """For cl_method='ours' the writer must pull from
    ``after_stage_<s>/reserved/test/...`` (the rank-truncated state, which
    is what carries forward to subsequent stages), not from
    ``after_stage_<s>/full_rank/test/...``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        args = _fake_args(tmp)
        args.cl_method = 'ours'
        stage_task_indices = [[0, 1], [2]]
        task_keys = ['IHM', 'LOS', 'BIRADS']
        task_slugs = ['ihm', 'los', 'birads']
        # Populate BOTH phases with deliberately different values so we can
        # tell which one the writer chose.
        cl_log = {}
        for slug, full_auc, reserved_auc in [
            ('ihm', 0.79, 0.75),
            ('los', 0.82, 0.79),
        ]:
            cl_log[f'after_stage_0/full_rank/test/{slug}/primary'] = full_auc
            cl_log[f'after_stage_0/full_rank/test/{slug}/auc'] = full_auc
            cl_log[f'after_stage_0/reserved/test/{slug}/primary'] = reserved_auc
            cl_log[f'after_stage_0/reserved/test/{slug}/auc'] = reserved_auc
        out_fname = _write_best_model_results_for_stage(
            args, s_idx=0, stage_label='ihm-los',
            stage_task_indices=stage_task_indices,
            task_keys=task_keys, task_slugs=task_slugs, cl_log=cl_log,
        )
        parsed = parse_best_block(out_fname)
        assert parsed is not None
        # ours -> reserved values (0.75 / 0.79), NOT full_rank (0.79 / 0.82).
        assert abs(parsed[0]['auc'] - 0.75) < 1e-6, parsed[0]
        assert abs(parsed[1]['auc'] - 0.79) < 1e-6, parsed[1]
        # Header file should also identify the phase in setting line.
        with open(out_fname) as f:
            content = f.read()
        assert 'phase=reserved' in content, content[:500]
        print(f'  cl_method=ours wrote reserved values, phase=reserved tagged in setting  [OK]')


def main():
    print('CL best_model_results writer compatibility tests')
    print('=' * 60)
    print('writer output parses back via aggregate_results.parse_best_block')
    print('-' * 60)
    test_writer_output_parses_back()
    print('\ntwo-stage runs produce distinct stage/seed paths')
    print('-' * 60)
    test_two_stages_separate_paths()
    print('\ncl_method=ours pulls reserved-phase values, not full_rank')
    print('-' * 60)
    test_ours_pulls_reserved_phase()
    print('\nAll best_model_results writer tests passed.')


if __name__ == '__main__':
    main()
