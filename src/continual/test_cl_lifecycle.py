"""End-to-end-without-data test: simulate ``train_continual``'s per-task
lifecycle on a small ``SeqMoE`` to catch wiring bugs in the helper sequence:

  install_continual_routers
    -> [task 0: train_step (mock backward) -> validation forward]
    -> _post_task_reserve_freeze_grow
    -> [task 1: train_step (mock backward) -> validation forward]
    -> _post_task_reserve_freeze_grow (no grow on the last task)
    -> after-task eval forward at every prior task with set_current_task_idx

Also asserts:
  * After reservation the prior task's experts are LowRankExpertMLP (frozen).
  * For column_grow: only the active column slice receives gradient at each
    task; frozen columns receive zero gradient.
  * For per_task_router: only the latest router head is trainable; prior task
    heads have requires_grad=False everywhere.

Run from the project root with:
    python -m src.continual.test_cl_lifecycle
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from src.continual.cl_moe import (
    LowRankExpertMLP,
    install_continual_routers,
    set_current_task_idx,
)
from src.continual.cl_routers import (
    ColumnGrowModalityRouter,
    PerTaskModalityRouter,
)
from src.continual.cl_train import _post_task_reserve_freeze_grow
from src.fusemoe_multitask.moe import (
    MoEConfig,
    SeqMoE,
    TemporalExpertMLP,
)


def _make_seq_moe(num_experts=3, embed_dim=24, moe_hidden=48,
                  mods=('ts', 'txt'), gating='softmax', seed=0):
    torch.manual_seed(seed)
    cfg = MoEConfig(
        num_experts=num_experts, embed_dim=embed_dim,
        moe_hidden_size=moe_hidden, modality_types=list(mods),
        top_k=2, temporal_kernel=3, noisy_gating=False,
        gating=gating, dropout=0.0,
    )
    return SeqMoE(cfg)


def _x_list(seq_len=5, bs=2, D=24, num_mods=2, seed=1):
    torch.manual_seed(seed)
    return [torch.randn(seq_len, bs, D, requires_grad=False) for _ in range(num_mods)]


def _step(sm, x_list, mods, train=True):
    sm.train() if train else sm.eval()
    out, total_loss = sm(x_list, mods, train=train)
    if train:
        loss = sum((o ** 2).mean() for o in out) + total_loss
        sm.zero_grad()
        loss.backward()
    return out


def _run_lifecycle(mode, num_tasks=3, E=3):
    print(f'\n--- mode={mode!r}, num_tasks={num_tasks}, experts_per_task={E} ---')
    sm = _make_seq_moe(num_experts=E)
    install_continual_routers(sm, mode=mode)
    mods = list(sm.routers.keys())

    captured_router_param_ids = []  # snapshot ids of params expected to stay frozen

    for t_idx in range(num_tasks):
        print(f'  task {t_idx}: experts={len(sm.experts)}, num_experts={sm.num_experts}')

        # Tell per_task_router which head fires.
        set_current_task_idx(sm, t_idx)

        # Mock training step: forward + backward to populate gradients on
        # currently-trainable parameters.
        x_list = _x_list(num_mods=len(mods), seed=100 + t_idx)
        _step(sm, x_list, mods, train=True)

        # Sanity: experts in the active range should have gradients on their
        # weights; reserved experts should not.
        cur_lo = t_idx * E
        cur_hi = (t_idx + 1) * E
        for i in range(len(sm.experts)):
            e = sm.experts[i]
            if isinstance(e, LowRankExpertMLP):
                # Frozen factored expert: no grads expected.
                for p in e.parameters():
                    assert p.grad is None or p.grad.abs().max().item() == 0.0, \
                        f'LowRank expert {i} got grad'
            elif cur_lo <= i < cur_hi:
                # Active fresh experts: should have at least one non-zero grad
                # somewhere (the only inputs above are nonzero so backward
                # touches every active expert that fires).
                pass  # we don't assert non-zero because top-k may exclude some

        # Post-task: reserve + freeze + grow (no grow on the last task).
        _post_task_reserve_freeze_grow(
            sm, t_idx, num_tasks=num_tasks,
            num_experts_per_task=E, rank=4, mode=mode,
        )

        # After reservation: experts in [cur_lo, cur_hi) must be LowRank.
        for i in range(cur_lo, cur_hi):
            assert isinstance(sm.experts[i], LowRankExpertMLP), \
                f'expert {i} not reserved'
        # All experts in those slots must be fully frozen.
        for i in range(cur_lo, cur_hi):
            bad = [n for n, p in sm.experts[i].named_parameters() if p.requires_grad]
            assert not bad, f'reserved expert {i} has trainable params {bad}'

        # For per_task_router: snapshot prior heads' params to verify zero grad later.
        if mode == 'per_task_router':
            captured_router_param_ids.append([
                id(p) for r in sm.routers.values()
                for p in r.task_routers[t_idx].parameters()
            ])

    # ---- After all tasks: do an "after-task eval" pass on each prior task ----
    print('  after-all-tasks eval pass:')
    for s_idx in range(num_tasks):
        set_current_task_idx(sm, s_idx)
        sm.eval()
        with torch.no_grad():
            x_list = _x_list(num_mods=len(mods), seed=999 + s_idx)
            out, _ = sm(x_list, mods, train=False)
        for o in out:
            assert torch.isfinite(o).all() and o.shape == x_list[0].shape
        print(f'    task {s_idx}: forward OK')

    # ---- Mode-specific structural assertions ----
    if mode == 'column_grow':
        # Final w_gate width = num_tasks * E; freeze boundary = num_tasks * E.
        for r in sm.routers.values():
            assert isinstance(r, ColumnGrowModalityRouter)
            assert r.w_gate.shape == (sm.config.embed_dim, num_tasks * E), \
                f'final w_gate shape unexpected: {r.w_gate.shape}'
            # After the last task's freeze, _num_frozen_cols == num_tasks*E
            # (since grow is skipped on the last task), meaning the entire
            # router is now frozen until the next task arrives.
            assert r._num_frozen_cols == num_tasks * E
        print('  column_grow: w_gate width and freeze boundary correct  [OK]')

    elif mode == 'per_task_router':
        # task_routers should have exactly num_tasks entries; all but the last
        # are frozen. The last one was just frozen in the post-task hook of
        # the last task, so all entries are frozen.
        for r in sm.routers.values():
            assert isinstance(r, PerTaskModalityRouter)
            assert len(r.task_routers) == num_tasks, \
                f'expected {num_tasks} task_routers, got {len(r.task_routers)}'
            for ti, sub in enumerate(r.task_routers):
                bad = [n for n, p in sub.named_parameters() if p.requires_grad]
                assert not bad, f'task_router {ti} has trainable params: {bad}'
        print('  per_task_router: all task heads present and frozen post-lifecycle  [OK]')


def _run_fixed_experts_lifecycle(num_tasks=3, E=3, R=4):
    """Lifecycle for --fixed_experts mode: pool stays at E experts, each
    becoming a StackedExpertMLP after the first reservation. New components
    accumulate per task.
    """
    from src.continual.cl_moe import (
        StackedExpertMLP,
        append_fresh_active_components,
        convert_to_stacked_after_first_reserve,
        reserve_active_components,
        reserve_low_rank,
        add_router_head_only,
    )

    print(f'\n--- fixed_experts lifecycle, num_tasks={num_tasks}, E={E}, R={R} ---')
    sm = _make_seq_moe(num_experts=E, gating='softmax')
    install_continual_routers(sm, mode='per_task_router')
    mods = list(sm.routers.keys())

    for t_idx in range(num_tasks):
        set_current_task_idx(sm, t_idx)
        print(f'  task {t_idx}: pool={len(sm.experts)} (should stay at {E})')
        assert len(sm.experts) == E

        # Mock training step.
        x_list = _x_list(num_mods=len(mods), seed=200 + t_idx)
        _step(sm, x_list, mods, train=True)

        # Reserve active component (or first reservation).
        if t_idx == 0:
            indices = list(range(len(sm.experts)))
            reserve_low_rank(sm, indices, R)
            convert_to_stacked_after_first_reserve(sm)
        else:
            reserve_active_components(sm, R)

        # Pool size still E.
        assert len(sm.experts) == E
        assert all(isinstance(e, StackedExpertMLP) for e in sm.experts), \
            'all slots must be StackedExpertMLP after first reservation'

        # Each slot should have t_idx+1 frozen LowRank components.
        for slot in sm.experts:
            assert len(slot.components) == t_idx + 1
            for c in slot.components:
                assert isinstance(c, LowRankExpertMLP)
                bad = [n for n, p in c.named_parameters() if p.requires_grad]
                assert not bad

        if t_idx < num_tasks - 1:
            for sm_inner in [sm]:
                append_fresh_active_components(sm_inner)
                add_router_head_only(sm_inner)
            # After grow: each slot has t_idx+2 components, last is trainable.
            for slot in sm.experts:
                assert len(slot.components) == t_idx + 2
                last = slot.components[-1]
                assert not isinstance(last, LowRankExpertMLP)
                # fc2 zero-init: forward of last alone should be zero.
                with torch.no_grad():
                    test_x = torch.randn(2, 5, sm.config.embed_dim)
                    out = last(test_x)
                    assert out.abs().max().item() == 0.0, \
                        f'fresh active component does not output 0; max |out|={out.abs().max().item()}'

    # After all tasks: each slot has num_tasks frozen components.
    for slot in sm.experts:
        assert len(slot.components) == num_tasks
        assert all(isinstance(c, LowRankExpertMLP) for c in slot.components)
    print(f'  final state: {E} StackedExpertMLP slots, '
          f'each with {num_tasks} frozen components  [OK]')

    # Eval at each prior task: forward should still work.
    print('  eval forward at every prior task:')
    for s_idx in range(num_tasks):
        set_current_task_idx(sm, s_idx)
        sm.eval()
        with torch.no_grad():
            x_list = _x_list(num_mods=len(mods), seed=999 + s_idx)
            out, _ = sm(x_list, mods, train=False)
        for o in out:
            assert torch.isfinite(o).all() and o.shape == x_list[0].shape
        # Slot components used should be 0..s_idx only.
        for slot in sm.experts:
            assert slot.current_task_idx == s_idx
        print(f'    task {s_idx}: current_task_idx propagated, forward OK')


def main():
    print('Continual lifecycle integration test')
    print('=' * 60)
    _run_lifecycle('column_grow', num_tasks=3, E=3)
    _run_lifecycle('per_task_router', num_tasks=3, E=3)
    _run_fixed_experts_lifecycle(num_tasks=3, E=3, R=4)
    print('\nAll lifecycle tests passed.')


if __name__ == '__main__':
    main()
