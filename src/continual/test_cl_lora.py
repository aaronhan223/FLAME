"""Unit tests for the LoRA continual-learning baseline (--cl_method lora).

Covers:
  1. LoRAExpertMLP with no adapters or zero-init adapters (B=0) produces
     forward output identical to the base TemporalExpertMLP alone.
  2. With current_task_idx slicing, only adapters [0..current-1] contribute
     to the effective weights.
  3. After randomizing B, forward differs from the base; the difference
     equals running the base with W += A@B for each weight matrix.
  4. Freezing semantics: only the latest LoRA adapter receives gradients;
     base + earlier adapters stay frozen during a backward pass.

Run from project root with:
    python -m src.continual.test_cl_lora
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from src.continual.cl_moe import (
    LoRAAdapter,
    LoRAConv1dK1Wrapper,
    LoRAExpertMLP,
    LoRALinearWrapper,
    append_fresh_encoder_lora_adapters,
    append_lora_adapter,
    convert_encoder_layers_to_lora,
    convert_to_lora_after_stage_0,
    freeze_active_encoder_lora_adapters,
    freeze_active_lora_adapter,
    set_current_task_idx,
    set_current_task_idx_for_encoder_lora,
)
from src.fusemoe_multitask.moe import TemporalExpertMLP


def _make_base(D=64, H=128, ks=3, seed=0):
    torch.manual_seed(seed)
    return TemporalExpertMLP(
        input_size=D, hidden_size=H, temporal_kernel=ks,
        dropout=0.0, hidden_act='gelu',
    ).eval()


def test_zero_adapters_match_base():
    """LoRAExpertMLP with no adapters (current_task_idx=0) must produce
    forward output bit-equal to the base TemporalExpertMLP."""
    base = _make_base()
    base_copy = copy.deepcopy(base).eval()
    lora = LoRAExpertMLP(base).eval()
    lora.current_task_idx = 0
    x = torch.randn(2, 5, 64)
    with torch.no_grad():
        out_base = base_copy(x)
        out_lora = lora(x)
    diff = (out_base - out_lora).abs().max().item()
    print(f'  zero adapters, current=0: max |diff| = {diff:.3e}')
    assert torch.allclose(out_base, out_lora, atol=1e-6), diff
    print('  [OK]')


def test_zero_init_adapter_matches_base():
    """A freshly-appended LoRAAdapter has B=0, so its delta is exactly 0
    and forward equals the base alone."""
    base = _make_base(seed=1)
    base_copy = copy.deepcopy(base).eval()
    lora = LoRAExpertMLP(base).eval()
    lora.append_adapter(lora_rank=4)
    lora.current_task_idx = 1  # base + 1 adapter (zero-init)

    x = torch.randn(2, 5, 64)
    with torch.no_grad():
        out_base = base_copy(x)
        out_lora = lora(x)
    diff = (out_base - out_lora).abs().max().item()
    print(f'  zero-init adapter (B=0), current=1: max |diff| = {diff:.3e}')
    assert torch.allclose(out_base, out_lora, atol=1e-6), diff
    print('  [OK]')


def test_current_task_idx_slicing():
    """Setting current_task_idx to a smaller value must mask out later
    adapters. With non-zero adapters, current=0 should differ from
    current=1 should differ from current=2."""
    base = _make_base(seed=2)
    lora = LoRAExpertMLP(base).eval()
    lora.append_adapter(lora_rank=4)
    lora.append_adapter(lora_rank=4)
    # Randomize B for both adapters so they have non-zero delta.
    torch.manual_seed(10)
    with torch.no_grad():
        for ad in lora.lora_adapters:
            ad.B_conv.copy_(torch.randn_like(ad.B_conv) * 0.1)
            ad.B_fc1.copy_(torch.randn_like(ad.B_fc1) * 0.1)
            ad.B_fc2.copy_(torch.randn_like(ad.B_fc2) * 0.1)

    x = torch.randn(2, 5, 64)
    outs = []
    with torch.no_grad():
        for c in (0, 1, 2):
            lora.current_task_idx = c
            outs.append(lora(x))

    diff_01 = (outs[0] - outs[1]).abs().max().item()
    diff_12 = (outs[1] - outs[2]).abs().max().item()
    diff_02 = (outs[0] - outs[2]).abs().max().item()
    print(f'  current=0 vs current=1: max |diff| = {diff_01:.3e}')
    print(f'  current=1 vs current=2: max |diff| = {diff_12:.3e}')
    print(f'  current=0 vs current=2: max |diff| = {diff_02:.3e}')
    assert diff_01 > 1e-5, 'adapter 0 has no effect!'
    assert diff_12 > 1e-5, 'adapter 1 has no effect!'
    assert diff_02 > diff_01, 'cumulative effect should be larger'
    print('  [OK]')


def test_effective_weights_match_explicit_addition():
    """Reconstruct W_eff manually via base + Σ A·B; compare against running
    a copy of the base with weights replaced by W_eff. The two must produce
    bit-equal forward output (modulo float ordering)."""
    base = _make_base(seed=3)
    lora = LoRAExpertMLP(base).eval()
    lora.append_adapter(lora_rank=4)
    torch.manual_seed(20)
    with torch.no_grad():
        ad = lora.lora_adapters[0]
        ad.B_conv.copy_(torch.randn_like(ad.B_conv) * 0.1)
        ad.B_fc1.copy_(torch.randn_like(ad.B_fc1) * 0.1)
        ad.B_fc2.copy_(torch.randn_like(ad.B_fc2) * 0.1)
    lora.current_task_idx = 1

    # Manual: build a TemporalExpertMLP whose weights = base + delta.
    manual = copy.deepcopy(base).eval()
    with torch.no_grad():
        manual.temporal_conv.weight.copy_(
            base.temporal_conv.weight + ad.delta_conv()
        )
        manual.fc1.weight.copy_(base.fc1.weight + ad.delta_fc1())
        manual.fc2.weight.copy_(base.fc2.weight + ad.delta_fc2())

    x = torch.randn(2, 5, 64)
    with torch.no_grad():
        out_lora = lora(x)
        out_manual = manual(x)
    diff = (out_lora - out_manual).abs().max().item()
    print(f'  W_eff equivalence: max |out_lora - out_manual| = {diff:.3e}')
    assert torch.allclose(out_lora, out_manual, atol=1e-5), diff
    print('  [OK]')


def test_only_active_adapter_receives_gradient():
    """When a LoRAExpertMLP has 2 adapters, with adapter[0] frozen and
    adapter[1] trainable, a backward through the slot must produce zero
    gradient on base + adapter[0], non-zero on adapter[1]."""
    base = _make_base(seed=4)
    lora = LoRAExpertMLP(base)
    # Stage 1: append + freeze first adapter (simulating end-of-stage-1 state).
    lora.append_adapter(lora_rank=4)
    for p in lora.lora_adapters[0].parameters():
        p.requires_grad = False
    # Stage 2: append second adapter (trainable).
    lora.append_adapter(lora_rank=4)
    lora.current_task_idx = 2
    # Base must already be frozen (it was frozen at convert_to_lora_after_stage_0).
    for p in lora.base.parameters():
        p.requires_grad = False

    x = torch.randn(2, 5, 64)
    out = lora(x)
    loss = (out ** 2).mean()
    loss.backward()

    # Base: no grad allowed.
    for n, p in lora.base.named_parameters():
        assert p.grad is None or p.grad.abs().max().item() == 0.0, \
            f'base.{n} got nonzero grad'
    # Adapter 0 (frozen): no grad allowed.
    for n, p in lora.lora_adapters[0].named_parameters():
        assert p.grad is None or p.grad.abs().max().item() == 0.0, \
            f'frozen adapter 0.{n} got nonzero grad'
    # Adapter 1 (active): must have nonzero grad somewhere.
    has_grad = any(
        p.grad is not None and p.grad.abs().max().item() > 0.0
        for p in lora.lora_adapters[1].parameters()
    )
    assert has_grad, 'active adapter 1 received no gradient'
    print('  base + adapter[0] frozen no grad; adapter[1] active has grad  [OK]')


def test_set_current_task_idx_propagates():
    """``set_current_task_idx`` should reach into nested LoRAExpertMLP
    instances (e.g., when they live inside a model)."""
    base = _make_base(seed=5)
    lora = LoRAExpertMLP(base)
    container = torch.nn.ModuleList([lora])
    set_current_task_idx(container, 7)
    assert lora.current_task_idx == 7
    print('  set_current_task_idx propagates to LoRAExpertMLP  [OK]')


class _EncToy(torch.nn.Module):
    """Toy encoder with a Linear, a Conv1d(k=1), a Conv1d(k=3) (out of scope),
    and a LayerNorm (out of scope). Used by the encoder-LoRA tests below.
    """

    def __init__(self, D=16):
        super().__init__()
        self.fc1 = torch.nn.Linear(D, D)
        self.conv1d_k1 = torch.nn.Conv1d(D, D, kernel_size=1)
        self.conv1d_k3 = torch.nn.Conv1d(D, D, kernel_size=3, padding=1)
        self.norm = torch.nn.LayerNorm(D)

    def forward(self, x):
        h = self.fc1(x)
        h = self.norm(h)
        h_c = self.conv1d_k1(h.transpose(1, 2)).transpose(1, 2)
        h_c = self.conv1d_k3(h_c.transpose(1, 2)).transpose(1, 2)
        return h_c


def test_encoder_lora_wraps_only_trainable_layers():
    torch.manual_seed(0)
    enc = _EncToy()
    # Freeze fc1 to confirm wrap is skipped (mimics frozen pretrained pieces).
    for p in enc.fc1.parameters():
        p.requires_grad = False

    counts = convert_encoder_layers_to_lora({'X': enc, 'Y': enc})  # shared dedup
    print(f'  wrap counts: {counts}')

    assert isinstance(enc.fc1, torch.nn.Linear), 'frozen fc1 should not be wrapped'
    assert isinstance(enc.conv1d_k1, LoRAConv1dK1Wrapper), \
        f'conv1d_k1 should be wrapped, got {type(enc.conv1d_k1).__name__}'
    assert isinstance(enc.conv1d_k3, torch.nn.Conv1d), \
        'conv1d_k3 (k=3) should NOT be wrapped'
    assert isinstance(enc.norm, torch.nn.LayerNorm), 'LayerNorm not wrapped'

    # Wrapped base must be frozen.
    assert all(not p.requires_grad for p in enc.conv1d_k1.base.parameters())
    print('  conv1d_k1 wrapped, frozen base; fc1/conv1d_k3/LayerNorm intact  [OK]')


def test_encoder_lora_zero_init_preserves_forward():
    torch.manual_seed(0)
    enc = _EncToy()
    convert_encoder_layers_to_lora({'X': enc})
    set_current_task_idx_for_encoder_lora({'X': enc}, 0)
    x = torch.randn(2, 7, 16)
    pre = enc(x).detach().clone()

    counts = append_fresh_encoder_lora_adapters({'X': enc}, rank=4)
    print(f'  append counts: {counts}')

    set_current_task_idx_for_encoder_lora({'X': enc}, 1)
    post = enc(x).detach()
    diff = (pre - post).abs().max().item()
    print(f'  pre vs post-append max |diff| = {diff:.2e}')
    assert diff < 1e-6, f'append introduced output drift {diff} (B should be 0)'

    assert isinstance(enc.fc1, LoRALinearWrapper)
    assert len(enc.fc1.lora_adapters) == 1
    assert isinstance(enc.conv1d_k1, LoRAConv1dK1Wrapper)
    assert len(enc.conv1d_k1.lora_adapters) == 1
    print('  fresh encoder LoRA adapters are zero-init -> output unchanged  [OK]')


def test_encoder_lora_only_active_adapter_trains():
    torch.manual_seed(0)
    enc = _EncToy()
    convert_encoder_layers_to_lora({'X': enc})
    append_fresh_encoder_lora_adapters({'X': enc}, rank=4)
    # Freeze "stage 1" by simulating end-of-stage freeze, then append next.
    freeze_active_encoder_lora_adapters({'X': enc})
    append_fresh_encoder_lora_adapters({'X': enc}, rank=4)

    # Now: base = frozen, adapters[0] = frozen, adapters[1] = trainable.
    set_current_task_idx_for_encoder_lora({'X': enc}, 2)
    x = torch.randn(2, 7, 16, requires_grad=False)
    out = enc(x).sum()
    out.backward()

    fc1 = enc.fc1
    assert fc1.base.weight.grad is None, 'base should not receive grads'
    # Frozen adapter[0] should have no grads (params have requires_grad=False).
    for p in fc1.lora_adapters[0].parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, \
            'frozen adapter should not receive grads'
    # Active adapter[1]: B is zero-initialised, so dL/dA = (dL/dW) @ B^T = 0
    # at initial step. Gradient flows to B first (dL/dB = A^T @ (dL/dW)).
    g_B = fc1.lora_adapters[1].B.grad
    assert g_B is not None and g_B.abs().sum() > 0, \
        'active adapter B should receive grads (A.grad is 0 because B starts at 0)'
    print('  only the active (last) adapter receives gradient (via B)  [OK]')


def test_encoder_lora_set_current_task_idx_slicing():
    torch.manual_seed(0)
    enc = _EncToy()
    convert_encoder_layers_to_lora({'X': enc})
    append_fresh_encoder_lora_adapters({'X': enc}, rank=4)
    # Make adapter[0]'s B non-zero so it actually contributes.
    with torch.no_grad():
        enc.fc1.lora_adapters[0].B.fill_(0.05)
    freeze_active_encoder_lora_adapters({'X': enc})
    append_fresh_encoder_lora_adapters({'X': enc}, rank=4)
    with torch.no_grad():
        enc.fc1.lora_adapters[1].B.fill_(0.10)

    x = torch.randn(2, 7, 16)
    set_current_task_idx_for_encoder_lora({'X': enc}, 0)
    out0 = enc(x).detach()  # base only
    set_current_task_idx_for_encoder_lora({'X': enc}, 1)
    out1 = enc(x).detach()  # base + adapters[0]
    set_current_task_idx_for_encoder_lora({'X': enc}, 2)
    out2 = enc(x).detach()  # base + adapters[0..1]

    d01 = (out0 - out1).abs().max().item()
    d12 = (out1 - out2).abs().max().item()
    print(f'  out0 vs out1 max |diff| = {d01:.2e}; out1 vs out2 max |diff| = {d12:.2e}')
    assert d01 > 1e-4, 'adapter 0 should change forward at idx=1'
    assert d12 > 1e-4, 'adapter 1 should change forward at idx=2'
    print('  current_task_idx slicing on encoder LoRA works  [OK]')


def main():
    print('LoRA continual-learning baseline tests')
    print('=' * 60)
    print('zero adapters -> base forward')
    print('-' * 60)
    test_zero_adapters_match_base()
    print('\nzero-init B adapter -> base forward')
    print('-' * 60)
    test_zero_init_adapter_matches_base()
    print('\ncurrent_task_idx slicing of adapters')
    print('-' * 60)
    test_current_task_idx_slicing()
    print('\neffective W = base + Σ ΔW manually verified')
    print('-' * 60)
    test_effective_weights_match_explicit_addition()
    print('\nonly the active adapter receives gradient')
    print('-' * 60)
    test_only_active_adapter_receives_gradient()
    print('\nset_current_task_idx propagates to LoRAExpertMLP')
    print('-' * 60)
    test_set_current_task_idx_propagates()

    print('\nencoder LoRA: wraps only trainable Linear / Conv1d(k=1)')
    print('-' * 60)
    test_encoder_lora_wraps_only_trainable_layers()

    print('\nencoder LoRA: zero-init B preserves forward at append')
    print('-' * 60)
    test_encoder_lora_zero_init_preserves_forward()

    print('\nencoder LoRA: only active adapter receives gradient')
    print('-' * 60)
    test_encoder_lora_only_active_adapter_trains()

    print('\nencoder LoRA: current_task_idx slicing affects forward')
    print('-' * 60)
    test_encoder_lora_set_current_task_idx_slicing()

    print('\nAll LoRA tests passed.')


if __name__ == '__main__':
    main()
