"""Unit tests for the EWC continual-learning baseline (--cl_method ewc).

Covers:
  1. ``iter_expert_weights`` selects only the three weight matrices we
     intend EWC to regularize -- nothing else (no biases, no LayerNorm,
     no router/encoder/head params).
  2. ``EWCState`` initial state has zero Fisher and an
     ``older_params`` snapshot that matches the model's expert weights,
     so the regularizer evaluates to exactly zero before any training.
  3. ``regularizer(model)`` grows quadratically with the displacement of
     a single expert weight from its snapshot.
  4. ``regularizer`` is zero on parameters where Fisher is zero
     (regardless of how far they've drifted).
  5. ``merge_fisher`` with alpha=0.5 yields the midpoint of old and
     current; alpha=1.0 keeps old; alpha=0.0 takes current.
  6. ``snapshot_older_params`` updates the anchor in place; subsequent
     regularizer calls reference the new anchor.

Run from project root:
    python -m src.continual.test_cl_ewc
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from src.continual.cl_ewc import (
    EWCState,
    iter_expert_and_router_weights,
    iter_expert_weights,
    iter_router_weights,
)
from src.continual.cl_moe import find_seq_moes, install_continual_routers
from src.continual.cl_routers import PerTaskModalityRouter
from src.fusemoe_multitask.moe import MoEConfig, SeqMoE


class _Layer(torch.nn.Module):
    def __init__(self, seq_moe):
        super().__init__()
        self.moe = seq_moe


class _MoEWrapper(torch.nn.Module):
    """Wraps a ``SeqMoE`` two levels deep so ``named_parameters`` yields
    names like ``layer.moe.experts.0.fc1.weight`` -- containing the
    substring ``.moe.experts.`` that ``iter_expert_weights`` filters on
    (with the leading dot). Mirrors the production layout
    ``MULTCrossModel.trans_self_cross_*.layers[k].moe.experts.*``.
    """

    def __init__(self, seq_moe):
        super().__init__()
        self.layer = _Layer(seq_moe)


def _make_seq_moe(num_experts=3, embed_dim=32, hidden=64, ks=3, seed=0):
    torch.manual_seed(seed)
    cfg = MoEConfig(
        num_experts=num_experts, embed_dim=embed_dim,
        moe_hidden_size=hidden, modality_types=['ts', 'txt'],
        top_k=2, temporal_kernel=ks, noisy_gating=False,
        gating='softmax', dropout=0.0,
    )
    return _MoEWrapper(SeqMoE(cfg)).eval()


def test_iter_expert_weights_selects_correct_subset():
    """Only temporal_conv.weight, fc1.weight, fc2.weight from each expert
    in each SeqMoE should be returned. No biases, no LayerNorm, no
    router/temporal_pool weights."""
    sm = _make_seq_moe(num_experts=3)
    pairs = list(iter_expert_weights(sm))
    names = [n for n, _ in pairs]
    print(f'  selected {len(names)} expert-weight params:')
    for n in names:
        print(f'    {n}')

    # Expect exactly 3 (matrices) * 3 (experts) = 9 entries on this SeqMoE.
    assert len(names) == 9, f'expected 9 entries, got {len(names)}'

    # Every name must contain `.moe.experts.` and end with one of the
    # three target suffixes.
    for n in names:
        assert '.moe.experts.' in n
        assert any(n.endswith(s) for s in (
            '.temporal_conv.weight', '.fc1.weight', '.fc2.weight'
        )), f'unexpected suffix: {n}'

    # Negative: no biases or LN params should appear.
    for n in names:
        assert not n.endswith('.bias'), f'bias leaked: {n}'
        assert 'temporal_norm' not in n, f'LN leaked: {n}'

    # Negative: no router params (w_gate, w_noise, temporal_pool.query).
    for n in names:
        assert 'w_gate' not in n and 'w_noise' not in n
        assert 'temporal_pool' not in n

    print('  [OK]')


def test_initial_state_zero_fisher_zero_regularizer():
    sm = _make_seq_moe(seed=1)
    ewc = EWCState(sm)

    # Initial Fisher must be zero for every tracked parameter.
    for n, f in ewc.fisher.items():
        assert f.abs().max().item() == 0.0, f'Fisher[{n}] not zero'

    # older_params equals current expert weights -> regularizer is zero.
    reg = ewc.regularizer(sm)
    print(f'  initial regularizer: {reg.item():.6e}')
    assert reg.item() == 0.0, f'initial regularizer != 0: {reg.item()}'
    print('  [OK]')


def test_regularizer_quadratic_in_displacement():
    """Set one expert weight matrix's Fisher to ones and displace it by
    delta. Regularizer should equal 0.5 * sum(delta^2). Doubling delta
    should quadruple the regularizer."""
    sm = _make_seq_moe(seed=2)
    ewc = EWCState(sm)

    # Pick one expert's fc1 weight: set Fisher to all-ones (so reg = 0.5 * ||Δ||²).
    target_name = None
    for n in ewc.fisher:
        if n.endswith('.fc1.weight'):
            target_name = n
            break
    assert target_name is not None
    ewc.fisher[target_name].fill_(1.0)

    # Displace by a known delta on that single weight matrix.
    target_param = dict(sm.named_parameters())[target_name]
    delta = 0.1
    with torch.no_grad():
        target_param.add_(delta)

    reg = ewc.regularizer(sm).item()
    expected = 0.5 * target_param.numel() * (delta ** 2)
    print(f'  reg @ delta=0.1: {reg:.6f}, expected {expected:.6f}')
    assert abs(reg - expected) < 1e-4, (reg, expected)

    # Double the displacement -> reg should quadruple.
    with torch.no_grad():
        target_param.add_(delta)  # now total displacement = 2*delta
    reg2 = ewc.regularizer(sm).item()
    expected2 = 0.5 * target_param.numel() * ((2 * delta) ** 2)
    print(f'  reg @ delta=0.2: {reg2:.6f}, expected {expected2:.6f} (=4x)')
    assert abs(reg2 - expected2) < 1e-3, (reg2, expected2)
    assert abs(reg2 / max(reg, 1e-12) - 4.0) < 1e-2

    print('  [OK]')


def test_regularizer_zero_on_zero_fisher_params():
    """Even with large displacement, params whose Fisher is zero contribute
    nothing to the regularizer."""
    sm = _make_seq_moe(seed=3)
    ewc = EWCState(sm)
    # Fisher stays zero for all params.

    # Move every tracked weight by a large amount.
    for n, p in iter_expert_weights(sm):
        with torch.no_grad():
            p.add_(1.0)

    reg = ewc.regularizer(sm).item()
    print(f'  reg with zero Fisher and large displacement: {reg:.6e}')
    assert reg == 0.0, f'expected 0 with zero Fisher, got {reg}'
    print('  [OK]')


def test_merge_fisher_alpha_semantics():
    """alpha=0.5 -> midpoint; alpha=1 -> keep old; alpha=0 -> take new."""
    sm = _make_seq_moe(seed=4)
    ewc = EWCState(sm)

    # Set self.fisher[*] = 4.0 and curr_fisher[*] = 8.0.
    for n in ewc.fisher:
        ewc.fisher[n].fill_(4.0)
    curr = {n: torch.full_like(f, 8.0) for n, f in ewc.fisher.items()}

    # alpha=0.5 -> midpoint = 6.0.
    ewc.merge_fisher(curr, alpha=0.5)
    for n, f in ewc.fisher.items():
        assert abs(f.mean().item() - 6.0) < 1e-6, (n, f.mean().item())

    # Reset and try alpha=1.0 -> Fisher unchanged from old (currently 6.0).
    for n in ewc.fisher:
        ewc.fisher[n].fill_(4.0)
    curr2 = {n: torch.full_like(f, 99.0) for n, f in ewc.fisher.items()}
    ewc.merge_fisher(curr2, alpha=1.0)
    for n, f in ewc.fisher.items():
        assert abs(f.mean().item() - 4.0) < 1e-6, (n, f.mean().item())

    # alpha=0.0 -> Fisher = curr_fisher (=99).
    ewc.merge_fisher(curr2, alpha=0.0)
    for n, f in ewc.fisher.items():
        assert abs(f.mean().item() - 99.0) < 1e-6, (n, f.mean().item())

    print('  alpha semantics: 0.5->mid, 1->old, 0->new  [OK]')


def test_snapshot_older_params_updates_anchor():
    """After moving expert weights and calling snapshot_older_params, the
    regularizer evaluated at the new (post-snapshot) state is again zero."""
    sm = _make_seq_moe(seed=5)
    ewc = EWCState(sm)

    # Set Fisher non-zero to make the regularizer responsive.
    for n in ewc.fisher:
        ewc.fisher[n].fill_(1.0)

    # Move expert weights, regularizer becomes positive.
    for _, p in iter_expert_weights(sm):
        with torch.no_grad():
            p.add_(0.5)
    reg_before = ewc.regularizer(sm).item()
    print(f'  reg after moving weights, before snapshot: {reg_before:.4f}')
    assert reg_before > 0

    # Snapshot the new state -> regularizer should evaluate to exactly 0
    # at this state.
    ewc.snapshot_older_params(sm)
    reg_after = ewc.regularizer(sm).item()
    print(f'  reg after snapshot (anchor updated): {reg_after:.6e}')
    assert reg_after == 0.0, reg_after

    # Move further -> regularizer becomes positive again.
    for _, p in iter_expert_weights(sm):
        with torch.no_grad():
            p.add_(0.5)
    reg_further = ewc.regularizer(sm).item()
    print(f'  reg after additional movement: {reg_further:.4f}')
    assert reg_further > 0

    print('  [OK]')


def test_regularizer_is_differentiable():
    """The regularizer must produce gradients on the model's expert
    weights (so it can be combined with the task loss in a single
    backward pass)."""
    sm = _make_seq_moe(seed=6)
    ewc = EWCState(sm)
    for n in ewc.fisher:
        ewc.fisher[n].fill_(1.0)
    # Move weights so the regularizer is nontrivial.
    for _, p in iter_expert_weights(sm):
        with torch.no_grad():
            p.add_(0.1)

    # Make sure expert params require grad.
    for _, p in iter_expert_weights(sm):
        p.requires_grad = True

    reg = ewc.regularizer(sm)
    sm.zero_grad()
    reg.backward()
    nonzero_grad = 0
    for n, p in iter_expert_weights(sm):
        if p.grad is not None and p.grad.abs().max().item() > 0:
            nonzero_grad += 1
    print(f'  expert weights with non-zero grad after backward: {nonzero_grad}')
    assert nonzero_grad > 0
    print('  [OK]')


def test_iter_router_weights_selects_correct_subset():
    """``iter_router_weights`` must yield only ``task_routers[0].w_gate`` and
    ``task_routers[0].w_noise`` on each ``PerTaskModalityRouter``. Other
    parameters (temporal_pool.query, mean/std buffers) must NOT appear,
    nor anything from ``task_routers[1+]`` if those exist."""
    sm = _make_seq_moe(num_experts=3)
    install_continual_routers(sm.layer.moe, mode='per_task_router')
    pairs = list(iter_router_weights(sm))
    names = [n for n, _ in pairs]
    print(f'  selected {len(names)} router-weight params:')
    for n in names:
        print(f'    {n}')

    # Expect 2 (w_gate + w_noise) per modality router (3 in our test:
    # 'ts', 'txt' + the default_router) = 6 entries (PerTaskModalityRouter
    # only adds task_routers[0] at install time).
    for n in names:
        assert '.task_routers.0.' in n
        assert n.endswith('.w_gate') or n.endswith('.w_noise')

    # Negative: no temporal_pool, no buffers.
    for n in names:
        assert 'temporal_pool' not in n
        assert 'mean' not in n.split('.')[-1] and 'std' not in n.split('.')[-1]

    # If we grow PerTaskModalityRouter (add task_routers[1]), those
    # weights must NOT be picked up by iter_router_weights.
    for r in sm.layer.moe.routers.values():
        if isinstance(r, PerTaskModalityRouter):
            r.add_task_router(sm.layer.moe.num_experts)
    if sm.layer.moe.default_router is not None:
        sm.layer.moe.default_router.add_task_router(sm.layer.moe.num_experts)

    names_after_grow = [n for n, _ in iter_router_weights(sm)]
    assert all('.task_routers.0.' in n for n in names_after_grow), names_after_grow
    assert all('.task_routers.1.' not in n for n in names_after_grow), names_after_grow
    print('  task_routers[1+] weights correctly excluded after grow  [OK]')


def test_iter_expert_and_router_weights_unions_both():
    """``iter_expert_and_router_weights`` should yield the union: every
    expert weight first, then every router weight."""
    sm = _make_seq_moe(num_experts=3)
    install_continual_routers(sm.layer.moe, mode='per_task_router')

    expert_names = [n for n, _ in iter_expert_weights(sm)]
    router_names = [n for n, _ in iter_router_weights(sm)]
    union_names = [n for n, _ in iter_expert_and_router_weights(sm)]

    assert union_names == expert_names + router_names, (
        f'union order mismatch:\n  union={union_names}\n  '
        f'expected={expert_names + router_names}'
    )
    print(f'  union: {len(expert_names)} expert + {len(router_names)} router '
          f'= {len(union_names)} entries  [OK]')


def test_ewc_state_with_combined_iterator():
    """EWCState constructed with ``iter_expert_and_router_weights`` must
    track BOTH expert and router weights in ``older_params`` and
    ``fisher``, and its regularizer must respond to drift in either."""
    sm = _make_seq_moe(num_experts=3)
    install_continual_routers(sm.layer.moe, mode='per_task_router')
    ewc = EWCState(sm, target_iterator=iter_expert_and_router_weights)

    # Set Fisher to ones for ALL tracked params (experts AND router).
    for n in ewc.fisher:
        ewc.fisher[n].fill_(1.0)

    # Move ONE expert weight; regularizer should be positive.
    expert_name = next(n for n in ewc.older_params if 'experts' in n)
    expert_param = dict(sm.named_parameters())[expert_name]
    with torch.no_grad():
        expert_param.add_(0.1)
    reg_after_expert_drift = ewc.regularizer(sm).item()
    print(f'  reg after expert drift only: {reg_after_expert_drift:.4f}')
    assert reg_after_expert_drift > 0

    # Move ONE router weight too; regularizer should grow.
    router_name = next(n for n in ewc.older_params if 'task_routers.0' in n)
    router_param = dict(sm.named_parameters())[router_name]
    with torch.no_grad():
        router_param.add_(0.1)
    reg_after_both = ewc.regularizer(sm).item()
    print(f'  reg after expert + router drift: {reg_after_both:.4f}')
    assert reg_after_both > reg_after_expert_drift, (
        f'regularizer did not grow when router drifted: '
        f'before={reg_after_expert_drift}, after={reg_after_both}'
    )
    print('  [OK]')


def main():
    print('EWC continual-learning baseline tests')
    print('=' * 60)
    print('iter_expert_weights selects correct subset')
    print('-' * 60)
    test_iter_expert_weights_selects_correct_subset()
    print('\ninitial state -> Fisher=0 and regularizer=0')
    print('-' * 60)
    test_initial_state_zero_fisher_zero_regularizer()
    print('\nregularizer is quadratic in displacement')
    print('-' * 60)
    test_regularizer_quadratic_in_displacement()
    print('\nregularizer is zero where Fisher is zero')
    print('-' * 60)
    test_regularizer_zero_on_zero_fisher_params()
    print('\nmerge_fisher alpha semantics')
    print('-' * 60)
    test_merge_fisher_alpha_semantics()
    print('\nsnapshot_older_params updates anchor in place')
    print('-' * 60)
    test_snapshot_older_params_updates_anchor()
    print('\nregularizer flows gradients back to expert weights')
    print('-' * 60)
    test_regularizer_is_differentiable()
    print('\niter_router_weights selects task_routers[0].{w_gate,w_noise}')
    print('-' * 60)
    test_iter_router_weights_selects_correct_subset()
    print('\niter_expert_and_router_weights unions both sets')
    print('-' * 60)
    test_iter_expert_and_router_weights_unions_both()
    print('\nEWCState with combined iterator covers experts + router')
    print('-' * 60)
    test_ewc_state_with_combined_iterator()
    print('\nAll EWC tests passed.')


if __name__ == '__main__':
    main()
