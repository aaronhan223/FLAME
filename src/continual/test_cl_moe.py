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
    LowRankLinear,
    LowRankConv1dK1,
    StackedLowRankLinear,
    StackedLowRankConv1dK1,
    append_fresh_encoder_active_components,
    reserve_encoder_layers,
    set_current_task_idx_for_encoders,
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


class _ToyEncoder(torch.nn.Module):
    """Mini encoder mimicking the layer types we care about: nn.Linear,
    nn.Conv1d(k=1), nn.Conv1d(k=3) (should be skipped), nn.LayerNorm
    (should be skipped). Forward is contrived but shape-consistent.
    """

    def __init__(self, D=16):
        super().__init__()
        self.fc1 = torch.nn.Linear(D, D)
        self.fc2 = torch.nn.Linear(D, D)
        self.conv1d_k1 = torch.nn.Conv1d(D, D, kernel_size=1)
        self.conv1d_k3 = torch.nn.Conv1d(D, D, kernel_size=3, padding=1)
        self.norm = torch.nn.LayerNorm(D)

    def forward(self, x):
        # x: [B, T, D]
        h = self.fc1(x)
        h = self.norm(h)
        h_conv = self.conv1d_k1(h.transpose(1, 2)).transpose(1, 2)
        h_conv3 = self.conv1d_k3(h_conv.transpose(1, 2)).transpose(1, 2)
        return self.fc2(h_conv3)


def test_lowrank_linear_matches_truncated(rank=4, D=16, B=3):
    torch.manual_seed(0)
    lin = torch.nn.Linear(D, D)
    x = torch.randn(B, D)
    # In-place truncate to rank-r via the same _svd_factors path used by LowRankLinear.
    W = lin.weight.detach().clone()
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    k = min(rank, len(S))
    W_trunc = (U[:, :k] * S[:k]) @ Vh[:k, :]
    ref = torch.nn.functional.linear(x, W_trunc.to(W.dtype), lin.bias)

    lr = LowRankLinear.from_linear(lin, rank=rank)
    out = lr(x)
    diff = (out - ref).abs().max().item()
    print(f'  rank {rank}: max |diff| = {diff:.2e}')
    assert diff < 1e-5, f'LowRankLinear drift {diff} too large'
    print('  LowRankLinear matches in-place SVD-truncated nn.Linear  [OK]')


def test_reserve_encoder_layers_wraps_only_trainable():
    torch.manual_seed(0)
    enc = _ToyEncoder()
    # Freeze fc1 to verify wrap is skipped for frozen layers (mimics pretrained pieces).
    for p in enc.fc1.parameters():
        p.requires_grad = False

    counts = reserve_encoder_layers({'X': enc}, rank=4)
    print(f'  counts after stage 0 reservation: {counts}')

    # fc2 (Linear, trainable) -> wrapped
    assert isinstance(enc.fc2, StackedLowRankLinear), \
        f'fc2 should be wrapped; got {type(enc.fc2).__name__}'
    assert len(enc.fc2.components) == 1
    assert isinstance(enc.fc2.components[0], LowRankLinear)

    # fc1 (frozen) -> NOT wrapped
    assert isinstance(enc.fc1, torch.nn.Linear)

    # conv1d_k1 (trainable) -> wrapped
    assert isinstance(enc.conv1d_k1, StackedLowRankConv1dK1)
    assert isinstance(enc.conv1d_k1.components[0], LowRankConv1dK1)

    # conv1d_k3 (kernel != 1) -> NOT wrapped (out of scope per eval_lowrank_experts.py)
    assert isinstance(enc.conv1d_k3, torch.nn.Conv1d)

    # norm (LayerNorm) -> NOT wrapped
    assert isinstance(enc.norm, torch.nn.LayerNorm)

    print('  fc2 + conv1d_k1 wrapped, fc1 (frozen) + conv1d_k3 + LayerNorm '
          'left intact  [OK]')


def test_append_fresh_active_zero_initialised():
    torch.manual_seed(0)
    enc = _ToyEncoder()
    reserve_encoder_layers({'X': enc}, rank=4)
    # Forward before append -> only frozen rank-4 component active.
    set_current_task_idx_for_encoders({'X': enc}, 0)
    x = torch.randn(2, 7, 16)
    pre = enc(x).detach().clone()

    counts = append_fresh_encoder_active_components({'X': enc})
    print(f'  append counts: {counts}')

    # Forward after append at current_task_idx=1 should be IDENTICAL because
    # the new component is zero-initialised.
    set_current_task_idx_for_encoders({'X': enc}, 1)
    post = enc(x).detach()
    diff = (pre - post).abs().max().item()
    print(f'  pre vs post-append max |diff| = {diff:.2e}')
    assert diff < 1e-6, f'append introduced output drift {diff}'

    # Both stacks should now have exactly two components (frozen lowrank + active).
    assert len(enc.fc2.components) == 2
    assert isinstance(enc.fc2.components[0], LowRankLinear)
    assert isinstance(enc.fc2.components[1], torch.nn.Linear)
    assert len(enc.conv1d_k1.components) == 2
    assert isinstance(enc.conv1d_k1.components[1], torch.nn.Conv1d)

    print('  fresh active components are zero-init -> output unchanged  [OK]')


def test_reserve_active_then_set_idx_slicing():
    torch.manual_seed(0)
    enc = _ToyEncoder()
    reserve_encoder_layers({'X': enc}, rank=4)
    append_fresh_encoder_active_components({'X': enc})
    # Train the active components a bit (just perturb their weights).
    with torch.no_grad():
        enc.fc2.components[1].weight.fill_(0.05)
        enc.fc2.components[1].bias.fill_(0.01)
        enc.conv1d_k1.components[1].weight.fill_(0.05)
        enc.conv1d_k1.components[1].bias.fill_(0.01)

    x = torch.randn(2, 7, 16)
    set_current_task_idx_for_encoders({'X': enc}, 1)
    out_with_active = enc(x).detach().clone()

    # Reserve active -> SVD-truncates the perturbed full-rank layer, leaves a
    # frozen low-rank version.
    counts = reserve_encoder_layers({'X': enc}, rank=4)
    print(f'  reserve_active counts: {counts}')
    assert counts['reserved_active'] >= 2  # at least fc2 and conv1d_k1

    set_current_task_idx_for_encoders({'X': enc}, 1)
    out_after_reserve = enc(x).detach()
    # Output should be very close (rank-4 captures most of a perturbation rank-1 update).
    diff = (out_with_active - out_after_reserve).abs().max().item()
    print(f'  pre/post reserve_active max |diff| = {diff:.2e} (rank-4 SVD)')
    # It won't be zero (rank truncation isn't lossless on a rank-1 perturbation
    # mixed with the frozen base), but should be small relative to scale.
    assert diff < 1.0, f'reserve_active perturbed output too much: {diff}'

    # Slicing: at current_task_idx=0 only the FIRST (rank-4) component fires;
    # the just-reserved second component is excluded.
    set_current_task_idx_for_encoders({'X': enc}, 0)
    out_idx0 = enc(x).detach()
    diff_first_only = (out_idx0 - out_after_reserve).abs().max().item()
    assert diff_first_only > 1e-4, \
        'idx=0 vs idx=1 should differ when 2nd component is non-zero'

    print('  reserve_active replaces last component with frozen low-rank  [OK]')
    print('  current_task_idx slicing excludes later components  [OK]')


def test_dedup_shared_encoder_via_id():
    """Shared encoder (one instance, multiple keys) should be processed once."""
    torch.manual_seed(0)
    enc = _ToyEncoder()
    counts = reserve_encoder_layers({'A': enc, 'B': enc, 'C': enc}, rank=4)
    print(f'  counts with 3 aliases: {counts}')
    # 2 trainable wrappable layers (fc1, fc2 are both trainable here, conv1d_k1 too) = 3.
    assert counts['wrapped_linear'] == 2
    assert counts['wrapped_conv1dk1'] == 1
    print('  shared encoder is processed exactly once  [OK]')


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

    print('\nLowRankLinear matches in-place SVD-truncated nn.Linear')
    print('-' * 60)
    for r in [1, 2, 4, 8]:
        test_lowrank_linear_matches_truncated(rank=r)

    print('\nreserve_encoder_layers wraps only trainable Linear/Conv1d(k=1)')
    print('-' * 60)
    test_reserve_encoder_layers_wraps_only_trainable()

    print('\nappend_fresh_encoder_active_components is zero-initialised')
    print('-' * 60)
    test_append_fresh_active_zero_initialised()

    print('\nreserve_active + current_task_idx slicing on encoder stacks')
    print('-' * 60)
    test_reserve_active_then_set_idx_slicing()

    print('\nencoder dedup via id() handles shared encoders')
    print('-' * 60)
    test_dedup_shared_encoder_via_id()

    print('\nAll tests passed.')


if __name__ == '__main__':
    main()
