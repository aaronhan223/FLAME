"""Unit tests for src.continual.cl_moe.

  1. Forward equivalence: ``LowRankExpertMLP.from_temporal_expert(expert, rank)``
     produces forward outputs identical (within float tolerance) to running
     ``expert`` after applying in-place rank-k SVD truncation to its three
     weight matrices.
  2. Storage savings: factored param count for low ranks is meaningfully
     less than the full-weight param count.

Run from the project root with:
    python -m src.continual.test_cl_moe
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from src.continual.cl_moe import (
    LowRankExpertMLP,
    svd_truncate_2d,
    svd_truncate_conv1d_weight,
)
from src.fusemoe_multitask.moe import TemporalExpertMLP


def _truncate_inplace(expert, rank):
    """Apply rank-k SVD truncation in-place to a TemporalExpertMLP. Mirrors
    ``apply_lowrank_to_experts`` in ``src/analysis/eval_lowrank_experts.py``.
    """
    with torch.no_grad():
        expert.temporal_conv.weight.copy_(
            svd_truncate_conv1d_weight(expert.temporal_conv.weight, rank)
        )
        expert.fc1.weight.copy_(svd_truncate_2d(expert.fc1.weight, rank))
        expert.fc2.weight.copy_(svd_truncate_2d(expert.fc2.weight, rank))


def test_forward_matches_truncated_full(rank, D=128, H=512, ks=3, B=4, T=10,
                                         atol=1e-4, rtol=1e-4):
    torch.manual_seed(0)
    src = TemporalExpertMLP(
        input_size=D, hidden_size=H, temporal_kernel=ks,
        dropout=0.1, hidden_act='gelu',
    )
    src.eval()

    src_truncated = copy.deepcopy(src)
    _truncate_inplace(src_truncated, rank)
    src_truncated.eval()

    lr = LowRankExpertMLP.from_temporal_expert(src, rank)
    lr.eval()

    x = torch.randn(B, T, D)
    with torch.no_grad():
        out_a = src_truncated(x)
        out_b = lr(x)

    diff = (out_a - out_b).abs().max().item()
    ok = torch.allclose(out_a, out_b, atol=atol, rtol=rtol)
    status = 'OK' if ok else 'FAIL'
    print(f'  rank={rank:>4}: max |out_inplace - out_factored| = {diff:.3e}  [{status}]')
    assert ok, (
        f'rank={rank}: max |Δ|={diff:.3e} exceeds tol (atol={atol}, rtol={rtol})'
    )


def test_zero_rank_produces_zero_weight_path(D=64, H=128, ks=3, B=2, T=5):
    """rank=0 stores zero factors; the conv path output is just bias broadcast,
    which after the residual + LayerNorm + zeroed MLP path equals the
    expected truncated-zero-weight forward."""
    torch.manual_seed(1)
    src = TemporalExpertMLP(
        input_size=D, hidden_size=H, temporal_kernel=ks,
        dropout=0.1, hidden_act='gelu',
    )
    src.eval()
    src_truncated = copy.deepcopy(src)
    _truncate_inplace(src_truncated, rank=0)
    src_truncated.eval()

    lr = LowRankExpertMLP.from_temporal_expert(src, rank=0)
    lr.eval()

    x = torch.randn(B, T, D)
    with torch.no_grad():
        out_a = src_truncated(x)
        out_b = lr(x)
    diff = (out_a - out_b).abs().max().item()
    ok = torch.allclose(out_a, out_b, atol=1e-5, rtol=1e-5)
    print(f'  rank=   0: max |out_inplace - out_factored| = {diff:.3e}  '
          f'[{"OK" if ok else "FAIL"}]')
    assert ok


def test_param_count(D=128, H=512, ks=3):
    torch.manual_seed(2)
    src = TemporalExpertMLP(
        input_size=D, hidden_size=H, temporal_kernel=ks,
        dropout=0.1, hidden_act='gelu',
    )
    full_w = sum(
        p.numel() for n, p in src.named_parameters()
        if n.endswith('.weight') and p.dim() >= 2
    )
    print(f'  Full weight-matrix param count: {full_w:,}')
    print(f'  rank | factored weight params | savings')
    for rank in [1, 4, 8, 16, 32, 64, 128]:
        lr = LowRankExpertMLP.from_temporal_expert(src, rank)
        fact = sum(
            p.numel() for n, p in lr.named_parameters()
            if any(s in n for s in ['_U', '_S', '_Vh'])
        )
        pct = 100.0 * fact / full_w
        print(f'  {rank:>4} | {fact:>10,}             | {pct:5.1f}% of full')


def test_all_params_frozen():
    """Sanity: every parameter of the low-rank module must be requires_grad=False."""
    torch.manual_seed(3)
    src = TemporalExpertMLP(input_size=64, hidden_size=128, temporal_kernel=3,
                            dropout=0.1, hidden_act='gelu')
    lr = LowRankExpertMLP.from_temporal_expert(src, rank=8)
    bad = [n for n, p in lr.named_parameters() if p.requires_grad]
    print(f'  trainable params on frozen module: {bad if bad else "(none)"}'
          f'  [{"OK" if not bad else "FAIL"}]')
    assert not bad, f'These params are unexpectedly trainable: {bad}'


def test_stacked_expert_zero_init_starts_at_prior_output(D=64, H=128, ks=3, B=2, T=5):
    """A freshly-zero-init'd active component must start with output exactly 0,
    so the StackedExpertMLP forward equals the sum of prior frozen components
    at the start of the new task."""
    from src.continual.cl_moe import StackedExpertMLP, zero_init_fc2
    from src.fusemoe_multitask.moe import TemporalExpertMLP as TE

    torch.manual_seed(0)
    # Simulate two completed tasks: build a stacked slot with 2 frozen
    # rank-r LowRankExpertMLPs.
    e0 = TE(input_size=D, hidden_size=H, temporal_kernel=ks, dropout=0.0).eval()
    e1 = TE(input_size=D, hidden_size=H, temporal_kernel=ks, dropout=0.0).eval()
    lr0 = LowRankExpertMLP.from_temporal_expert(e0, rank=8).eval()
    lr1 = LowRankExpertMLP.from_temporal_expert(e1, rank=8).eval()

    s = StackedExpertMLP()
    s.components.append(lr0)
    s.components.append(lr1)
    s.eval()
    s.current_task_idx = 1  # use both prior components

    x = torch.randn(B, T, D)
    with torch.no_grad():
        prior_sum = lr0(x) + lr1(x)
        before_active = s(x)
    assert torch.allclose(before_active, prior_sum, atol=1e-5), 'stacked sum != lr0+lr1'

    # Append a fresh zero-init'd active component.
    e_active = zero_init_fc2(TE(input_size=D, hidden_size=H, temporal_kernel=ks,
                                 dropout=0.0)).eval()
    s.components.append(e_active)
    s.current_task_idx = 2
    with torch.no_grad():
        after_active = s(x)
        active_alone = e_active(x)
    print(f'  active_alone abs.max = {active_alone.abs().max().item():.3e}')
    assert active_alone.abs().max().item() == 0.0, (
        'zero-init fc2 did not produce zero output'
    )
    assert torch.allclose(after_active, prior_sum, atol=1e-5), (
        'StackedExpertMLP output changed after appending zero-init active'
    )
    print('  StackedExpertMLP zero-init: prior sum preserved on append  [OK]')


def test_stacked_expert_masking_via_current_task_idx(D=64, H=128, ks=3, B=2, T=5):
    """Set current_task_idx to a prior task; verify forward equals the partial
    sum over only the prior components (later components don't contribute)."""
    from src.continual.cl_moe import StackedExpertMLP
    from src.fusemoe_multitask.moe import TemporalExpertMLP as TE

    torch.manual_seed(1)
    es = [TE(input_size=D, hidden_size=H, temporal_kernel=ks, dropout=0.0).eval()
          for _ in range(3)]
    lrs = [LowRankExpertMLP.from_temporal_expert(e, rank=8).eval() for e in es]

    s = StackedExpertMLP()
    for lr in lrs:
        s.components.append(lr)
    s.eval()

    x = torch.randn(B, T, D)
    with torch.no_grad():
        partial_sums = [lrs[0](x), lrs[0](x) + lrs[1](x), lrs[0](x) + lrs[1](x) + lrs[2](x)]
        for k in range(3):
            s.current_task_idx = k
            out = s(x)
            diff = (out - partial_sums[k]).abs().max().item()
            print(f'  current_task_idx={k}: max |diff vs partial sum| = {diff:.3e}')
            assert torch.allclose(out, partial_sums[k], atol=1e-5)
    print('  StackedExpertMLP current_task_idx slicing matches partial sums  [OK]')


def test_stacked_reserve_active_replaces_with_lowrank(D=64, H=128, ks=3, B=2, T=5):
    """``reserve_active`` SVD-truncates the last component in place; forward
    after reservation must equal forward with the truncated approximation."""
    from src.continual.cl_moe import StackedExpertMLP, svd_truncate_2d, svd_truncate_conv1d_weight
    from src.fusemoe_multitask.moe import TemporalExpertMLP as TE
    import copy

    torch.manual_seed(2)
    e_prior = TE(input_size=D, hidden_size=H, temporal_kernel=ks, dropout=0.0).eval()
    lr_prior = LowRankExpertMLP.from_temporal_expert(e_prior, rank=8).eval()

    e_active = TE(input_size=D, hidden_size=H, temporal_kernel=ks, dropout=0.0).eval()
    s = StackedExpertMLP()
    s.components.append(lr_prior)
    s.components.append(e_active)
    s.eval()
    s.current_task_idx = 1

    x = torch.randn(B, T, D)

    # Reference: in-place SVD truncate a copy of e_active to rank 4 and forward.
    e_truncated = copy.deepcopy(e_active)
    with torch.no_grad():
        e_truncated.temporal_conv.weight.copy_(
            svd_truncate_conv1d_weight(e_truncated.temporal_conv.weight, 4)
        )
        e_truncated.fc1.weight.copy_(svd_truncate_2d(e_truncated.fc1.weight, 4))
        e_truncated.fc2.weight.copy_(svd_truncate_2d(e_truncated.fc2.weight, 4))
    e_truncated.eval()

    with torch.no_grad():
        ref = lr_prior(x) + e_truncated(x)

    s.reserve_active(rank=4)
    assert isinstance(s.components[-1], LowRankExpertMLP)
    bad = [n for n, p in s.components[-1].named_parameters() if p.requires_grad]
    assert not bad, f'reserved component still has trainable params: {bad}'

    with torch.no_grad():
        out = s(x)
    diff = (out - ref).abs().max().item()
    print(f'  reserve_active forward equivalence: max |diff| = {diff:.3e}')
    assert torch.allclose(out, ref, atol=1e-5)
    print('  reserve_active: forward matches manually SVD-truncated reference  [OK]')


def main():
    print('LowRankExpertMLP forward equivalence (eval mode)')
    print('-' * 60)
    for rank in [1, 4, 8, 16, 32, 64, 128]:
        test_forward_matches_truncated_full(rank=rank)

    print('\nrank=0 edge case')
    print('-' * 60)
    test_zero_rank_produces_zero_weight_path()

    print('\nAll-parameters-frozen check')
    print('-' * 60)
    test_all_params_frozen()

    print('\nStorage savings')
    print('-' * 60)
    test_param_count()

    print('\nStackedExpertMLP zero-init preserves prior sum')
    print('-' * 60)
    test_stacked_expert_zero_init_starts_at_prior_output()

    print('\nStackedExpertMLP current_task_idx slicing')
    print('-' * 60)
    test_stacked_expert_masking_via_current_task_idx()

    print('\nStackedExpertMLP reserve_active forward equivalence')
    print('-' * 60)
    test_stacked_reserve_active_replaces_with_lowrank()

    print('\nAll tests passed.')


if __name__ == '__main__':
    main()
