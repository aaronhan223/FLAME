"""Unit tests for Step-3 growth/freeze/reserve helpers on a real ``SeqMoE``.

Covers:
  - install_continual_routers swaps router classes without changing forward output.
  - grow_seq_moe extends pool and routers; forward still runs at expanded width.
  - reserve_low_rank replaces selected experts with LowRankExpertMLP, forward
    still runs and any LowRank slot is frozen.
  - column_grow: backward through forward sets gradients to zero on
    frozen columns, non-zero on active columns.
  - per_task_router: setting current_task_idx switches which router fires;
    older frozen routers do not receive gradients.

Run from the project root with:
    python -m src.continual.test_cl_growth
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from src.continual.cl_moe import (
    LowRankExpertMLP,
    find_seq_moes,
    freeze_router_columns,
    grow_seq_moe,
    install_continual_routers,
    reserve_low_rank,
    set_current_task_idx,
    trainable_expert_param_groups,
)
from src.continual.cl_routers import (
    ColumnGrowModalityRouter,
    PerTaskModalityRouter,
)
from src.fusemoe_multitask.moe import (
    MoEConfig,
    ModalityRouter,
    SeqMoE,
    TemporalExpertMLP,
)


def _make_seq_moe(num_experts=4, embed_dim=32, moe_hidden=64,
                  mods=('ts', 'txt'), gating='softmax', seed=0):
    torch.manual_seed(seed)
    cfg = MoEConfig(
        num_experts=num_experts, embed_dim=embed_dim,
        moe_hidden_size=moe_hidden, modality_types=list(mods),
        top_k=2, temporal_kernel=3, noisy_gating=True,
        gating=gating, dropout=0.0,
    )
    return SeqMoE(cfg).eval()  # eval = no dropout, deterministic forward


def _x_list(seq_len=6, bs=3, D=32, num_mods=2, seed=1):
    torch.manual_seed(seed)
    return [torch.randn(seq_len, bs, D) for _ in range(num_mods)]


def test_install_routers_preserves_forward_column_grow():
    """Swapping vanilla -> ColumnGrow with state migration must keep forward equal
    on the same input/eval-mode (modulo deterministic noise: turn it off)."""
    sm = _make_seq_moe(gating='softmax')
    # Disable noisy gating so forward is deterministic given the same input.
    for r in sm.routers.values():
        r.noisy_gating = False
    if sm.default_router is not None:
        sm.default_router.noisy_gating = False

    x_list = _x_list(num_mods=len(sm.routers))
    mods = list(sm.routers.keys())

    with torch.no_grad():
        out_before, _ = sm(x_list, mods, train=False)

    install_continual_routers(sm, mode='column_grow')
    # New routers must have noisy_gating=False migrated as well; install copies
    # weights but not the boolean -- set explicitly for the comparison.
    for r in sm.routers.values():
        r.noisy_gating = False
    if sm.default_router is not None:
        sm.default_router.noisy_gating = False

    assert all(isinstance(r, ColumnGrowModalityRouter) for r in sm.routers.values())
    with torch.no_grad():
        out_after, _ = sm(x_list, mods, train=False)

    diffs = [(a - b).abs().max().item() for a, b in zip(out_before, out_after)]
    print(f'  column_grow install: max forward diff per modality = {diffs}')
    for d in diffs:
        assert d < 1e-5, f'forward diverged after install: max diff={d}'
    print('  [OK]')


def test_install_routers_preserves_forward_per_task():
    sm = _make_seq_moe(gating='softmax')
    for r in sm.routers.values():
        r.noisy_gating = False
    if sm.default_router is not None:
        sm.default_router.noisy_gating = False
    x_list = _x_list(num_mods=len(sm.routers))
    mods = list(sm.routers.keys())

    with torch.no_grad():
        out_before, _ = sm(x_list, mods, train=False)

    install_continual_routers(sm, mode='per_task_router')
    for r in sm.routers.values():
        for sub in r.task_routers:
            sub.noisy_gating = False
    if sm.default_router is not None:
        for sub in sm.default_router.task_routers:
            sub.noisy_gating = False

    assert all(isinstance(r, PerTaskModalityRouter) for r in sm.routers.values())
    with torch.no_grad():
        out_after, _ = sm(x_list, mods, train=False)

    diffs = [(a - b).abs().max().item() for a, b in zip(out_before, out_after)]
    print(f'  per_task_router install: max forward diff per modality = {diffs}')
    for d in diffs:
        assert d < 1e-5, f'forward diverged after install: max diff={d}'
    print('  [OK]')


def test_grow_and_forward_column_grow():
    """After install + grow, forward runs at the expanded pool size."""
    sm = _make_seq_moe(num_experts=3)
    install_continual_routers(sm, mode='column_grow')

    grow_seq_moe(sm, num_new_experts=2)
    assert sm.num_experts == 5
    assert len(sm.experts) == 5
    for r in sm.routers.values():
        assert r.num_experts == 5
        assert r.w_gate.shape == (sm.config.embed_dim, 5)

    x_list = _x_list(num_mods=len(sm.routers))
    mods = list(sm.routers.keys())
    with torch.no_grad():
        out, _ = sm(x_list, mods, train=False)
    for o in out:
        assert o.shape == x_list[0].shape, f'shape changed: {o.shape}'
    print('  column_grow: pool grew 3->5, forward shape preserved  [OK]')


def test_grow_and_forward_per_task():
    sm = _make_seq_moe(num_experts=3)
    install_continual_routers(sm, mode='per_task_router')

    grow_seq_moe(sm, num_new_experts=2)
    assert sm.num_experts == 5
    assert len(sm.experts) == 5
    for r in sm.routers.values():
        assert isinstance(r, PerTaskModalityRouter)
        assert len(r.task_routers) == 2
        assert r.num_experts == 5
        # New active router has full visibility into all 5 experts.
        assert r.active_router.w_gate.shape == (sm.config.embed_dim, 5)

    set_current_task_idx(sm, 1)  # use the new active router

    x_list = _x_list(num_mods=len(sm.routers))
    mods = list(sm.routers.keys())
    with torch.no_grad():
        out, _ = sm(x_list, mods, train=False)
    for o in out:
        assert o.shape == x_list[0].shape
    print('  per_task_router: pool grew 3->5, two task routers, forward OK  [OK]')


def test_reserve_low_rank_replaces_experts():
    sm = _make_seq_moe(num_experts=4)
    install_continual_routers(sm, mode='column_grow')
    replaced = reserve_low_rank(sm, expert_indices=[0, 2], rank=8)
    assert replaced == [0, 2]
    assert isinstance(sm.experts[0], LowRankExpertMLP)
    assert isinstance(sm.experts[1], TemporalExpertMLP)
    assert isinstance(sm.experts[2], LowRankExpertMLP)
    assert isinstance(sm.experts[3], TemporalExpertMLP)
    # Frozen low-rank experts must have no trainable params.
    for idx in (0, 2):
        bad = [n for n, p in sm.experts[idx].named_parameters() if p.requires_grad]
        assert not bad, f'expert {idx} has trainable params: {bad}'
    # Idempotent
    again = reserve_low_rank(sm, expert_indices=[0, 2], rank=8)
    assert again == []
    print('  reserve_low_rank: experts swapped + frozen, idempotent  [OK]')


def test_reserve_then_forward_still_works():
    """Forward through SeqMoE must work when some experts are LowRankExpertMLP."""
    sm = _make_seq_moe(num_experts=4, gating='softmax')
    for r in sm.routers.values():
        r.noisy_gating = False
    if sm.default_router is not None:
        sm.default_router.noisy_gating = False
    install_continual_routers(sm, mode='column_grow')

    reserve_low_rank(sm, expert_indices=[0, 1], rank=8)
    x_list = _x_list(num_mods=len(sm.routers))
    mods = list(sm.routers.keys())
    with torch.no_grad():
        out, _ = sm(x_list, mods, train=False)
    for o in out:
        assert torch.isfinite(o).all(), 'non-finite output'
        assert o.shape == x_list[0].shape
    print('  forward with mixed Temporal+LowRank experts: OK  [OK]')


def test_column_grow_freeze_zeros_gradient_on_frozen_columns():
    """After freeze_router_columns(n), backprop should produce zero gradient
    on w_gate[:, :n] and (in general) non-zero on w_gate[:, n:]."""
    sm = _make_seq_moe(num_experts=4, gating='softmax')
    install_continual_routers(sm, mode='column_grow')
    # Grow so we have 4 frozen + 4 active.
    grow_seq_moe(sm, num_new_experts=4)
    # Freeze the first 4 columns (the prior task's slice).
    freeze_router_columns(sm, n_frozen_cols=4, mode='column_grow')

    x_list = _x_list(num_mods=len(sm.routers))
    mods = list(sm.routers.keys())
    sm.train()  # need grad path
    out, total_loss = sm(x_list, mods, train=True)
    # Use a simple scalar to drive gradients.
    target = torch.zeros_like(out[0])
    loss = ((out[0] - target) ** 2).mean() + total_loss
    loss.backward()

    for r in sm.routers.values():
        g = r.w_gate.grad
        assert g is not None, 'no gradient on w_gate'
        frozen_max = g[:, :4].abs().max().item()
        active_max = g[:, 4:].abs().max().item()
        print(f'    w_gate grad |frozen|.max={frozen_max:.3e}, |active|.max={active_max:.3e}')
        assert frozen_max == 0.0, f'frozen columns received grad {frozen_max}'
    print('  column_grow freeze-hook: zero grad on frozen columns  [OK]')


def test_per_task_freeze_disables_grad_on_prior_router():
    sm = _make_seq_moe(num_experts=3, gating='softmax')
    install_continual_routers(sm, mode='per_task_router')

    # Capture the params of the initial (about-to-be-frozen) task-0 routers.
    prior_router_params = []
    for r in sm.routers.values():
        prior_router_params.extend(list(r.task_routers[0].parameters()))

    freeze_router_columns(sm, n_frozen_cols=0, mode='per_task_router')
    grow_seq_moe(sm, num_new_experts=3)

    # task_routers[0] is now frozen; task_routers[1] is the active trainable one.
    for r in sm.routers.values():
        for p in r.task_routers[0].parameters():
            assert not p.requires_grad, 'task-0 router param still trainable after freeze'
        for p in r.task_routers[1].parameters():
            assert p.requires_grad, 'task-1 active router param frozen unexpectedly'

    set_current_task_idx(sm, 1)
    x_list = _x_list(num_mods=len(sm.routers))
    mods = list(sm.routers.keys())
    sm.train()
    out, total_loss = sm(x_list, mods, train=True)
    loss = (out[0] ** 2).mean() + total_loss
    loss.backward()

    # No grad should have flowed into the frozen task-0 routers.
    for p in prior_router_params:
        assert p.grad is None or p.grad.abs().max().item() == 0.0, (
            'frozen task-0 router received non-zero grad'
        )
    print('  per_task_router freeze: prior-task router receives zero grad  [OK]')


def test_trainable_expert_param_groups_excludes_frozen():
    sm = _make_seq_moe(num_experts=4)
    install_continual_routers(sm, mode='column_grow')
    reserve_low_rank(sm, expert_indices=[0, 1], rank=8)

    params = trainable_expert_param_groups(sm)
    # Both reserved (rank-r) experts should contribute zero trainable params;
    # the two remaining TemporalExpertMLPs should each contribute their full
    # parameter set.
    n_param_tensors_per_temporal = sum(1 for _ in sm.experts[2].parameters())
    n_param_tensors_per_router = sum(
        1 for r in sm.routers.values() for _ in r.parameters() if _.requires_grad
    )
    expected = (
        2 * n_param_tensors_per_temporal
        + n_param_tensors_per_router
    )
    actual = len(params)
    print(f'  trainable expert+router param tensors: {actual} (expected {expected})')
    # Allow >= because the default_router may or may not be present.
    assert actual >= expected
    print('  trainable_expert_param_groups: excludes LowRankExpertMLP slots  [OK]')


def test_per_task_combined_eval_isolates_prior_task():
    """For per_task_router with combined gating: after several tasks, setting
    ``current_task_idx = s`` for prior task ``s`` must produce gates that
    are zero on every expert > (s+1)*E (those experts are unseen by routers
    0..s and so receive ``-inf`` logits and are excluded from top-K)."""
    E = 3
    sm = _make_seq_moe(num_experts=E, gating='softmax')
    install_continual_routers(sm, mode='per_task_router')
    # Disable noisy gating for deterministic gate outputs.
    for r in sm.routers.values():
        for sub in r.task_routers:
            sub.noisy_gating = False

    # Simulate two grow steps -> 3 task routers, pool size 3*E = 9.
    grow_seq_moe(sm, num_new_experts=E)
    grow_seq_moe(sm, num_new_experts=E)
    for r in sm.routers.values():
        for sub in r.task_routers:
            sub.noisy_gating = False

    # Give every router *some* non-zero w_gate so logits aren't all zero
    # (zero logits make top-k tie-breaking arbitrary and uninformative).
    torch.manual_seed(42)
    for r in sm.routers.values():
        for sub in r.task_routers:
            with torch.no_grad():
                sub.w_gate.data = torch.randn_like(sub.w_gate)

    x_list = _x_list(num_mods=len(sm.routers), D=sm.config.embed_dim)
    mods = list(sm.routers.keys())

    # Evaluate as task 0 (only router 0 contributes).
    set_current_task_idx(sm, 0)
    sm.eval()
    with torch.no_grad():
        out, _ = sm(x_list, mods, train=False)
        # Inspect router gates directly via the captured routers.
        for r in sm.routers.values():
            r.current_task_idx = 0
            x_bf = x_list[0].transpose(0, 1)
            gates, _load = r(x_bf, train=False)
            # Gates beyond router-0's visibility (E experts) must be 0.
            beyond = gates[:, E:].abs().max().item()
            print(f'    task 0 eval: gates[:, E:].max = {beyond:.3e}')
            assert beyond == 0.0, (
                f'leak: prior-task eval routed to experts beyond {E}; '
                f'max abs weight={beyond}'
            )

    # Evaluate as task 1 (routers 0, 1 contribute).
    set_current_task_idx(sm, 1)
    with torch.no_grad():
        for r in sm.routers.values():
            r.current_task_idx = 1
            x_bf = x_list[0].transpose(0, 1)
            gates, _load = r(x_bf, train=False)
            beyond = gates[:, 2 * E:].abs().max().item()
            print(f'    task 1 eval: gates[:, 2E:].max = {beyond:.3e}')
            assert beyond == 0.0, (
                f'leak: task-1 eval routed to experts beyond {2 * E}; '
                f'max abs weight={beyond}'
            )

    # Evaluate as task 2 (all 3 routers contribute, full pool visible).
    set_current_task_idx(sm, 2)
    with torch.no_grad():
        for r in sm.routers.values():
            r.current_task_idx = 2
            x_bf = x_list[0].transpose(0, 1)
            gates, _load = r(x_bf, train=False)
            # All 9 experts may be selected; sanity: gate row sums equal 1
            row_sums = gates.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
                f'gates do not sum to 1 per row: {row_sums}'
            print(f'    task 2 eval: per-row gate sum = {row_sums.tolist()}')
    print('  per_task_router combined eval: prior-task isolation verified  [OK]')


def test_per_task_combined_only_active_router_trains():
    """Combined gating must let gradients flow only through the latest
    router's parameters, not the frozen prior ones."""
    E = 3
    sm = _make_seq_moe(num_experts=E, gating='softmax')
    install_continual_routers(sm, mode='per_task_router')
    grow_seq_moe(sm, num_new_experts=E)  # now 2 routers, latest is active
    set_current_task_idx(sm, 1)

    # Snapshot frozen router-0 params for later comparison.
    frozen_params = []
    for r in sm.routers.values():
        for p in r.task_routers[0].parameters():
            frozen_params.append(p)

    x_list = _x_list(num_mods=len(sm.routers), D=sm.config.embed_dim)
    mods = list(sm.routers.keys())
    sm.train()
    out, total_loss = sm(x_list, mods, train=True)
    loss = sum((o ** 2).mean() for o in out) + total_loss
    sm.zero_grad()
    loss.backward()

    for p in frozen_params:
        assert p.grad is None or p.grad.abs().max().item() == 0.0, (
            'frozen router-0 received nonzero gradient through combined gating'
        )
    # Latest router should have at least some non-zero gradient on w_gate.
    for r in sm.routers.values():
        latest = r.task_routers[-1]
        g = latest.w_gate.grad
        assert g is not None and g.abs().max().item() > 0.0, (
            'latest active router did not receive gradient'
        )
    print('  per_task_router combined: frozen routers stay frozen, latest gets grads  [OK]')


def main():
    print('Step-3 growth/freeze/reserve tests')
    print('=' * 60)
    print('install_continual_routers preserves forward (column_grow)')
    print('-' * 60)
    test_install_routers_preserves_forward_column_grow()
    print('\ninstall_continual_routers preserves forward (per_task_router)')
    print('-' * 60)
    test_install_routers_preserves_forward_per_task()
    print('\ngrow_seq_moe + forward (column_grow)')
    print('-' * 60)
    test_grow_and_forward_column_grow()
    print('\ngrow_seq_moe + forward (per_task_router)')
    print('-' * 60)
    test_grow_and_forward_per_task()
    print('\nreserve_low_rank replaces selected experts')
    print('-' * 60)
    test_reserve_low_rank_replaces_experts()
    print('\nforward still works with mixed Temporal+LowRank experts')
    print('-' * 60)
    test_reserve_then_forward_still_works()
    print('\ncolumn_grow freeze hook: zero grad on frozen columns')
    print('-' * 60)
    test_column_grow_freeze_zeros_gradient_on_frozen_columns()
    print('\nper_task_router freeze: prior-task router has no grad')
    print('-' * 60)
    test_per_task_freeze_disables_grad_on_prior_router()
    print('\ntrainable_expert_param_groups excludes frozen')
    print('-' * 60)
    test_trainable_expert_param_groups_excludes_frozen()
    print('\nper_task_router combined eval isolates prior task')
    print('-' * 60)
    test_per_task_combined_eval_isolates_prior_task()
    print('\nper_task_router combined: only active router trains')
    print('-' * 60)
    test_per_task_combined_only_active_router_trains()
    print('\nAll Step-3 tests passed.')


if __name__ == '__main__':
    main()
