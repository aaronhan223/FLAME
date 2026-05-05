"""Continual learning MoE primitives.

Provides:
  - ``svd_truncate_2d`` / ``svd_truncate_conv1d_weight``: utilities mirroring
    the rank-k truncation used in ``src/analysis/eval_lowrank_experts.py``.
  - ``LowRankExpertMLP``: a frozen, low-rank-factored drop-in replacement for
    ``TemporalExpertMLP`` (from ``src/fusemoe_multitask/moe.py``). Stores
    each weight matrix as factored ``(U, S, V_h)`` parameters with
    ``requires_grad=False``. Forward path reconstructs ``W = (U * S) @ V_h``
    and applies the same op (``F.conv1d`` / ``F.linear``) as the source
    expert, so its output is numerically equivalent to the source expert
    with its weights replaced by their rank-k SVD truncations.
  - ``LowRankExpertMLP.from_temporal_expert``: factory that builds an
    instance from a trained ``TemporalExpertMLP`` at a chosen rank.

Used by the continual pipeline to "reserve" each task's experts at the
end of its training stage. Step-2 only ships the module + tests; growth
and reservation logic land in Step-3.
"""

import math
from typing import Iterable, List

import torch
import torch.nn.functional as F
from torch import nn

from src.activations import ACT2FN
from src.fusemoe_multitask.moe import (
    ModalityRouter,
    SeqMoE,
    TemporalExpertMLP,
)
from src.continual.cl_routers import (
    ColumnGrowModalityRouter,
    PerTaskModalityRouter,
)


def _svd_factors(W, rank):
    """Return rank-k SVD factors ``(U[m,k], S[k], V_h[k,n])`` of a 2D matrix.

    Rank is clamped to ``min(W.shape)``. ``rank <= 0`` returns rank-1 zero
    factors so downstream shapes stay valid (forward output is identically 0).
    """
    if rank <= 0:
        m, n = W.shape
        return (
            torch.zeros(m, 1, device=W.device, dtype=W.dtype),
            torch.zeros(1, device=W.device, dtype=W.dtype),
            torch.zeros(1, n, device=W.device, dtype=W.dtype),
        )
    # Run SVD on CPU to dodge cuSolver init failures (CUSOLVER_STATUS_INTERNAL_ERROR
    # on cusolverDnCreate) seen after memory-hot fp16 training. Expert weight
    # matrices are tiny (<= a few hundred rows/cols) so CPU SVD is sub-ms.
    orig_device = W.device
    U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(W.detach().float().cpu(), full_matrices=False)
    k = min(rank, len(S_cpu))
    U = U_cpu[:, :k].to(device=orig_device, dtype=W.dtype)
    S = S_cpu[:k].to(device=orig_device, dtype=W.dtype)
    Vh = Vh_cpu[:k, :].to(device=orig_device, dtype=W.dtype)
    return U, S, Vh


def _reconstruct(U, S, Vh):
    """Reconstruct full matrix ``W = (U * S) @ V_h`` from factored form."""
    return (U * S) @ Vh


def svd_truncate_2d(W, rank):
    """Rank-k SVD truncation of a 2D matrix. Mirrors ``truncate_to_rank`` in
    ``src/analysis/eval_lowrank_experts.py``.
    """
    if rank <= 0:
        return torch.zeros_like(W)
    U, S, Vh = _svd_factors(W, rank)
    return _reconstruct(U, S, Vh)


def svd_truncate_conv1d_weight(W3d, rank):
    """Rank-k SVD truncation of a Conv1d weight ``[out, in, ks]``: reshape to
    2D ``[out, in*ks]``, truncate, reshape back.
    """
    out_c, in_c, ks = W3d.shape
    W2d = W3d.reshape(out_c, in_c * ks)
    return svd_truncate_2d(W2d, rank).reshape(out_c, in_c, ks)


class LowRankExpertMLP(nn.Module):
    """Frozen low-rank drop-in for ``TemporalExpertMLP``.

    The three weight matrices of the source expert are stored in factored
    SVD form. Biases and LayerNorm parameters are kept at full size (also
    frozen). Forward reconstructs each truncated weight and applies the
    same op as the source expert, producing identical output to running
    the source expert after in-place rank-k SVD truncation of its weights.

    Build via :py:meth:`from_temporal_expert` from a trained
    ``TemporalExpertMLP`` at the desired rank.
    """

    def __init__(self, input_size, hidden_size, temporal_kernel=3,
                 dropout=0.1, hidden_act='gelu', rank=16):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.temporal_kernel = temporal_kernel
        self.padding = temporal_kernel // 2
        self.rank_request = rank
        self.activation = ACT2FN[hidden_act]
        self.dropout = nn.Dropout(dropout)

        # temporal_conv weight reshaped to 2D: [input_size, input_size*ks]
        m_c, n_c = input_size, input_size * temporal_kernel
        k_c = max(1, min(rank, m_c, n_c))
        self.conv_U = nn.Parameter(torch.zeros(m_c, k_c), requires_grad=False)
        self.conv_S = nn.Parameter(torch.zeros(k_c), requires_grad=False)
        self.conv_Vh = nn.Parameter(torch.zeros(k_c, n_c), requires_grad=False)
        self.conv_bias = nn.Parameter(torch.zeros(input_size), requires_grad=False)
        self._k_conv = k_c

        # LayerNorm
        self.norm_weight = nn.Parameter(torch.ones(input_size), requires_grad=False)
        self.norm_bias = nn.Parameter(torch.zeros(input_size), requires_grad=False)

        # fc1 weight: [hidden_size, input_size]
        m1, n1 = hidden_size, input_size
        k1 = max(1, min(rank, m1, n1))
        self.fc1_U = nn.Parameter(torch.zeros(m1, k1), requires_grad=False)
        self.fc1_S = nn.Parameter(torch.zeros(k1), requires_grad=False)
        self.fc1_Vh = nn.Parameter(torch.zeros(k1, n1), requires_grad=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self._k_fc1 = k1

        # fc2 weight: [input_size, hidden_size]
        m2, n2 = input_size, hidden_size
        k2 = max(1, min(rank, m2, n2))
        self.fc2_U = nn.Parameter(torch.zeros(m2, k2), requires_grad=False)
        self.fc2_S = nn.Parameter(torch.zeros(k2), requires_grad=False)
        self.fc2_Vh = nn.Parameter(torch.zeros(k2, n2), requires_grad=False)
        self.fc2_bias = nn.Parameter(torch.zeros(input_size), requires_grad=False)
        self._k_fc2 = k2

    def _conv_weight(self):
        W2d = _reconstruct(self.conv_U, self.conv_S, self.conv_Vh)
        return W2d.reshape(self.input_size, self.input_size, self.temporal_kernel)

    def _fc1_weight(self):
        return _reconstruct(self.fc1_U, self.fc1_S, self.fc1_Vh)

    def _fc2_weight(self):
        return _reconstruct(self.fc2_U, self.fc2_S, self.fc2_Vh)

    def forward(self, x):
        # Mirror of TemporalExpertMLP.forward.
        x_t = x.transpose(1, 2)
        x_conv = F.conv1d(x_t, self._conv_weight(), bias=self.conv_bias,
                          padding=self.padding)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.activation(x_conv)
        h = F.layer_norm(x + x_conv, (self.input_size,),
                         self.norm_weight, self.norm_bias)

        out = F.linear(h, self._fc1_weight(), self.fc1_bias)
        out = self.activation(out)
        out = self.dropout(out)
        out = F.linear(out, self._fc2_weight(), self.fc2_bias)
        return out

    @classmethod
    def from_temporal_expert(cls, expert, rank):
        """Build a frozen rank-``rank`` factored copy of ``expert``."""
        input_size = expert.fc1.in_features
        hidden_size = expert.fc1.out_features
        kernel_size = expert.temporal_conv.kernel_size[0]
        dropout_p = expert.dropout.p if isinstance(expert.dropout, nn.Dropout) else 0.0

        # ``hidden_act='gelu'`` is just used to construct the dropout/activation
        # placeholder; we override ``activation`` below to whatever the source
        # expert actually used so behavior matches even for non-GELU experts.
        new = cls(
            input_size=input_size, hidden_size=hidden_size,
            temporal_kernel=kernel_size, dropout=dropout_p,
            hidden_act='gelu', rank=rank,
        )
        new.activation = expert.activation

        # Always use the requested ``rank`` (not the clamped buffer size) so
        # that ``rank <= 0`` uses ``_svd_factors``'s zero-factor branch instead
        # of producing the rank-1 SVD truncation by accident. The buffer sizes
        # in ``__init__`` are clamped to ``max(1, min(rank, m, n))``, which
        # matches what ``_svd_factors`` returns for any ``rank`` (including 0).
        with torch.no_grad():
            # temporal_conv: 3D -> 2D for SVD
            W_conv_3d = expert.temporal_conv.weight.detach()
            W_conv_2d = W_conv_3d.reshape(input_size, input_size * kernel_size)
            U, S, Vh = _svd_factors(W_conv_2d, rank)
            new.conv_U.copy_(U)
            new.conv_S.copy_(S)
            new.conv_Vh.copy_(Vh)
            new.conv_bias.copy_(expert.temporal_conv.bias.detach())

            new.norm_weight.copy_(expert.temporal_norm.weight.detach())
            new.norm_bias.copy_(expert.temporal_norm.bias.detach())

            U, S, Vh = _svd_factors(expert.fc1.weight.detach(), rank)
            new.fc1_U.copy_(U)
            new.fc1_S.copy_(S)
            new.fc1_Vh.copy_(Vh)
            new.fc1_bias.copy_(expert.fc1.bias.detach())

            U, S, Vh = _svd_factors(expert.fc2.weight.detach(), rank)
            new.fc2_U.copy_(U)
            new.fc2_S.copy_(S)
            new.fc2_Vh.copy_(Vh)
            new.fc2_bias.copy_(expert.fc2.bias.detach())

        for p in new.parameters():
            p.requires_grad = False
        return new


# =============================================================================
# Continual orchestration: walk the model, apply per-SeqMoE operations.
# =============================================================================


def find_seq_moes(model):
    """Yield every ``SeqMoE`` instance reachable from ``model``."""
    for module in model.modules():
        if isinstance(module, SeqMoE):
            yield module


def _migrate_router_state(src_router, dst_router):
    """Copy ``w_gate``, ``w_noise`` from ``src_router`` into ``dst_router``.

    Both are expected to share ``embed_dim`` and ``num_experts``. Used when
    swapping a vanilla ``ModalityRouter`` for one of the continual variants
    after model construction so we don't lose freshly-initialized values.
    """
    with torch.no_grad():
        if hasattr(dst_router, 'w_gate') and hasattr(src_router, 'w_gate'):
            dst_router.w_gate.data.copy_(src_router.w_gate.data)
        if hasattr(dst_router, 'w_noise') and hasattr(src_router, 'w_noise'):
            dst_router.w_noise.data.copy_(src_router.w_noise.data)


def install_continual_routers(model, mode='column_grow', combine='mean'):
    """Replace every vanilla ``ModalityRouter`` inside the model's ``SeqMoE``
    blocks with the corresponding continual variant.

    Done once, immediately after ``MULTCrossModel`` construction. Numerically
    equivalent at task 0 (state is migrated), and unlocks growth/freeze
    operations for tasks 1+. ``combine`` is the per-expert reduction operator
    used by ``PerTaskModalityRouter`` to merge logits across the routers
    visible at inference time (``'mean'`` / ``'sum'`` / ``'max'``); ignored
    for ``column_grow``.
    """
    if mode not in ('column_grow', 'per_task_router'):
        raise ValueError(f"Unknown router_growth_mode: {mode!r}")

    def _make_new(cfg, num_experts):
        if mode == 'column_grow':
            return ColumnGrowModalityRouter(
                embed_dim=cfg.embed_dim, num_experts=num_experts,
                top_k=cfg.top_k, gating=cfg.gating, noisy_gating=cfg.noisy_gating,
            )
        return PerTaskModalityRouter(
            embed_dim=cfg.embed_dim, num_experts=num_experts,
            top_k=cfg.top_k, gating=cfg.gating, noisy_gating=cfg.noisy_gating,
            combine=combine,
        )

    for sm in find_seq_moes(model):
        cfg = sm.config
        for mod_name in list(sm.routers.keys()):
            old = sm.routers[mod_name]
            new = _make_new(cfg, sm.num_experts).to(next(old.parameters()).device)
            if mode == 'column_grow':
                _migrate_router_state(old, new)
            else:
                _migrate_router_state(old, new.task_routers[0])
            sm.routers[mod_name] = new

        if sm.default_router is not None:
            old = sm.default_router
            new = _make_new(cfg, sm.num_experts).to(next(old.parameters()).device)
            if mode == 'column_grow':
                _migrate_router_state(old, new)
            else:
                _migrate_router_state(old, new.task_routers[0])
            sm.default_router = new
    return model


def _iter_routers(seq_moe):
    """Yield every router living on ``seq_moe`` (``routers`` values + default)."""
    for r in seq_moe.routers.values():
        yield r
    if seq_moe.default_router is not None:
        yield seq_moe.default_router


def grow_seq_moe(seq_moe, num_new_experts):
    """Append ``num_new_experts`` fresh ``TemporalExpertMLP``s to the pool, and
    grow every router on this ``SeqMoE`` to match the new pool size.

    For ``ColumnGrowModalityRouter`` this appends columns; for
    ``PerTaskModalityRouter`` this appends a fresh per-task router that sees
    every expert (frozen + new). Caller is responsible for invoking
    :py:func:`reserve_low_rank` on the previous task's experts and
    :py:func:`freeze_router_columns` *before* calling ``grow_seq_moe`` so the
    freeze boundary is up-to-date when the new task starts training.
    """
    if num_new_experts <= 0:
        return seq_moe
    cfg = seq_moe.config
    cur_n = len(seq_moe.experts)
    device = next(seq_moe.experts[0].parameters()).device
    new_total = cur_n + num_new_experts

    for i in range(num_new_experts):
        e = TemporalExpertMLP(
            input_size=cfg.embed_dim, hidden_size=cfg.moe_hidden_size,
            temporal_kernel=cfg.temporal_kernel, dropout=cfg.dropout,
            hidden_act=cfg.hidden_act, expert_idx=cur_n + i,
            num_experts=new_total,
        ).to(device)
        seq_moe.experts.append(e)

    seq_moe.num_experts = new_total
    seq_moe.config.num_experts = new_total

    for r in _iter_routers(seq_moe):
        if isinstance(r, ColumnGrowModalityRouter):
            r.grow(num_new_experts)
        elif isinstance(r, PerTaskModalityRouter):
            r.add_task_router(new_total)
        else:
            raise TypeError(
                f'Router {type(r).__name__} is not a continual variant; '
                'call install_continual_routers(model, mode=...) first.'
            )
    return seq_moe


def reserve_low_rank(seq_moe, expert_indices: Iterable[int], rank: int):
    """Replace experts at ``expert_indices`` in-place with frozen
    ``LowRankExpertMLP`` rank-``rank`` factored copies.

    Idempotent: experts that are already ``LowRankExpertMLP`` are skipped.
    Returns the list of indices that were replaced.
    """
    replaced = []
    for idx in sorted(set(int(i) for i in expert_indices)):
        original = seq_moe.experts[idx]
        if isinstance(original, LowRankExpertMLP):
            continue
        device = next(original.parameters()).device
        lr = LowRankExpertMLP.from_temporal_expert(original, rank).to(device)
        seq_moe.experts[idx] = lr
        replaced.append(idx)
    return replaced


def freeze_router_columns(seq_moe, n_frozen_cols: int, mode: str):
    """Freeze the first ``n_frozen_cols`` columns / the active task router.

    For ``column_grow``: sets ``_num_frozen_cols`` on every
    :class:`ColumnGrowModalityRouter` so subsequent backward passes will
    zero gradients in those columns.

    For ``per_task_router``: marks the currently-active router as frozen
    (its parameters get ``requires_grad=False``). Argument ``n_frozen_cols``
    is unused in this mode but kept for a uniform call signature.
    """
    for r in _iter_routers(seq_moe):
        if mode == 'column_grow':
            if isinstance(r, ColumnGrowModalityRouter):
                r.freeze_first(n_frozen_cols)
            else:
                raise TypeError(
                    f"Expected ColumnGrowModalityRouter, got {type(r).__name__}."
                )
        elif mode == 'per_task_router':
            if isinstance(r, PerTaskModalityRouter):
                r.freeze_active()
            else:
                raise TypeError(
                    f"Expected PerTaskModalityRouter, got {type(r).__name__}."
                )
        else:
            raise ValueError(f"Unknown mode: {mode!r}")


def set_current_task_idx(model, task_idx: int):
    """Set ``current_task_idx`` on every ``PerTaskModalityRouter``,
    ``StackedExpertMLP``, and ``LoRAExpertMLP`` in the model. The router
    uses it to pick which per-task head fires; the stacked expert uses it
    to slice the components list (sum components ``[0..task_idx]``); the
    LoRA expert uses it to select how many adapters are summed into the
    effective weights (``base + adapters[0..task_idx-1]``). No-op for
    plain modules."""
    for module in model.modules():
        if isinstance(module, PerTaskModalityRouter):
            module.current_task_idx = int(task_idx)
        elif isinstance(module, StackedExpertMLP):
            module.current_task_idx = int(task_idx)
        elif isinstance(module, LoRAExpertMLP):
            module.current_task_idx = int(task_idx)


def trainable_expert_param_groups(seq_moe):
    """Return the parameters that should be trainable on this ``SeqMoE`` for the
    *currently active* task: experts that are still ``TemporalExpertMLP``
    (i.e., not yet reserved as ``LowRankExpertMLP``) plus the active
    router parameters. Useful for building a per-task optimizer.
    """
    params: List[nn.Parameter] = []
    for e in seq_moe.experts:
        if not isinstance(e, LowRankExpertMLP):
            params.extend(p for p in e.parameters() if p.requires_grad)
    for r in _iter_routers(seq_moe):
        params.extend(p for p in r.parameters() if p.requires_grad)
    return params


# =============================================================================
# Fixed-experts (--fixed_experts) primitives.
# =============================================================================


class StackedExpertMLP(nn.Module):
    """Drop-in replacement for ``TemporalExpertMLP`` used in ``--fixed_experts``
    mode. Holds a stack of expert components, one per task that has trained
    against this slot. All components share the same input shape.

    Component lifecycle:
      * After a task finishes training, its component is a frozen
        ``LowRankExpertMLP`` (rank ``--reserved_rank``).
      * During the *current* task's training, the last entry is a
        trainable ``TemporalExpertMLP`` whose ``fc2.weight`` and
        ``fc2.bias`` are zero-initialised so the slot's net behaviour at
        the start of the task is identical to its post-reservation state
        from the previous task.

    Forward output is the sum of the first ``current_task_idx + 1``
    components. Setting ``current_task_idx = s`` reproduces the slot's
    behaviour as it was right after task ``s`` reservation, regardless of
    how many later tasks have since been added. The masking is *implicit*
    in the slice; no separate gating is needed.
    """

    def __init__(self):
        super().__init__()
        self.components = nn.ModuleList()
        self.current_task_idx = 0

    def append_active(self, fresh_expert):
        """Append a freshly-initialised ``TemporalExpertMLP`` as the new
        trainable component. Caller must ensure ``fresh_expert`` was
        constructed with the same ``input_size``/``hidden_size``/
        ``temporal_kernel`` as prior components.
        """
        self.components.append(fresh_expert)

    def reserve_active(self, rank):
        """Replace the last (active) component with its rank-``rank`` SVD
        truncation as a ``LowRankExpertMLP`` and freeze it. No-op if the
        last component is already a ``LowRankExpertMLP``.
        """
        if len(self.components) == 0:
            return
        last = self.components[-1]
        if isinstance(last, LowRankExpertMLP):
            return
        device = next(last.parameters()).device
        lr = LowRankExpertMLP.from_temporal_expert(last, rank).to(device)
        self.components[-1] = lr

    def forward(self, x):
        if len(self.components) == 0:
            return torch.zeros_like(x)
        n_use = min(int(self.current_task_idx) + 1, len(self.components))
        if n_use <= 0:
            n_use = 1
        out = self.components[0](x)
        for i in range(1, n_use):
            out = out + self.components[i](x)
        return out


def zero_init_fc2(temporal_expert):
    """Zero-initialise ``fc2.weight`` and ``fc2.bias`` on a ``TemporalExpertMLP``
    so its initial forward output is exactly zero. ``temporal_conv``,
    ``temporal_norm``, and ``fc1`` keep PyTorch's default random init;
    gradients still flow normally because ``dL/dfc2.weight = fc2_input *
    dL/dfc2_output`` is non-zero.
    """
    with torch.no_grad():
        temporal_expert.fc2.weight.zero_()
        if temporal_expert.fc2.bias is not None:
            temporal_expert.fc2.bias.zero_()
    return temporal_expert


def _make_fresh_expert(seq_moe, expert_idx, num_experts):
    """Build a freshly-initialised ``TemporalExpertMLP`` matching the shapes
    of the slot at index ``expert_idx`` on ``seq_moe``. Zero-inits ``fc2``
    so the new component contributes nothing at the start of the task.
    """
    cfg = seq_moe.config
    device = next(seq_moe.experts[0].parameters()).device
    e = TemporalExpertMLP(
        input_size=cfg.embed_dim, hidden_size=cfg.moe_hidden_size,
        temporal_kernel=cfg.temporal_kernel, dropout=cfg.dropout,
        hidden_act=cfg.hidden_act, expert_idx=expert_idx,
        num_experts=num_experts,
    ).to(device)
    return zero_init_fc2(e)


def convert_to_stacked_after_first_reserve(seq_moe):
    """After the very first call to ``reserve_low_rank`` on this ``SeqMoE``,
    wrap each ``LowRankExpertMLP`` slot in a ``StackedExpertMLP`` with one
    component (the just-reserved expert). Idempotent: slots already wrapped
    are skipped.
    """
    for i in range(len(seq_moe.experts)):
        e = seq_moe.experts[i]
        if isinstance(e, StackedExpertMLP):
            continue
        if not isinstance(e, LowRankExpertMLP):
            continue
        device = next(e.parameters()).device
        s = StackedExpertMLP().to(device)
        s.components.append(e)
        seq_moe.experts[i] = s


def append_fresh_active_components(seq_moe):
    """For each ``StackedExpertMLP`` slot, append a zero-initialised
    ``TemporalExpertMLP`` as the new trainable component for the upcoming
    task. Pool size on ``seq_moe`` is left unchanged.
    """
    for i in range(len(seq_moe.experts)):
        slot = seq_moe.experts[i]
        if not isinstance(slot, StackedExpertMLP):
            raise TypeError(
                f"slot {i} is {type(slot).__name__}, expected StackedExpertMLP. "
                "Did you forget to call convert_to_stacked_after_first_reserve?"
            )
        fresh = _make_fresh_expert(seq_moe, i, seq_moe.num_experts)
        slot.append_active(fresh)


def reserve_active_components(seq_moe, rank):
    """For each ``StackedExpertMLP`` slot, SVD-truncate the last (active)
    component to ``rank`` and replace it in place with a frozen
    ``LowRankExpertMLP``. No effect on slots whose last component is
    already low-rank (idempotent).
    """
    for i in range(len(seq_moe.experts)):
        slot = seq_moe.experts[i]
        if not isinstance(slot, StackedExpertMLP):
            continue
        slot.reserve_active(rank)


def add_router_head_only(seq_moe):
    """Add a new per-task router head to every ``PerTaskModalityRouter`` on
    this ``SeqMoE`` *without* growing the expert pool. New head's ``w_gate``
    is sized for ``seq_moe.num_experts`` (unchanged).
    """
    for r in _iter_routers(seq_moe):
        if isinstance(r, PerTaskModalityRouter):
            r.add_task_router(seq_moe.num_experts)
        else:
            raise TypeError(
                f"add_router_head_only requires PerTaskModalityRouter, got "
                f"{type(r).__name__}. --fixed_experts requires "
                "--router_growth_mode per_task_router."
            )


# =============================================================================
# LoRA continual-learning baseline (--cl_method lora).
# =============================================================================


class LoRAAdapter(nn.Module):
    """Three additive low-rank adapters, one each for ``temporal_conv``,
    ``fc1``, ``fc2`` of a ``TemporalExpertMLP``. Each adapter is a pair
    ``(A, B)`` of factored matrices with rank ``lora_rank``. The conv weight
    is treated as a 2D ``[D, D*ks]`` matrix for LoRA factorization and
    reshaped back to ``[D, D, ks]`` when emitting the delta.

    Init follows standard LoRA: ``A ~ kaiming_uniform_``, ``B = 0`` so the
    initial delta is exactly zero. Gradient still flows to ``A`` because
    ``dL/dA = (dL/d(AB)) @ B^T``, which is non-zero whenever ``dL/d(AB)``
    is — wait, that's zero when ``B = 0``. The initial gradient flows to
    ``B`` instead: ``dL/dB = A^T @ (dL/d(AB))``, which is non-zero when
    ``A`` has been kaiming-initialized.
    """

    def __init__(self, input_size, hidden_size, temporal_kernel, lora_rank):
        super().__init__()
        D, H, ks, r = input_size, hidden_size, temporal_kernel, lora_rank
        # temporal_conv weight reshaped 2D form: [D, D*ks].
        self.A_conv = nn.Parameter(torch.empty(D, r))
        self.B_conv = nn.Parameter(torch.zeros(r, D * ks))
        # fc1.weight: [H, D].
        self.A_fc1 = nn.Parameter(torch.empty(H, r))
        self.B_fc1 = nn.Parameter(torch.zeros(r, D))
        # fc2.weight: [D, H].
        self.A_fc2 = nn.Parameter(torch.empty(D, r))
        self.B_fc2 = nn.Parameter(torch.zeros(r, H))

        nn.init.kaiming_uniform_(self.A_conv, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A_fc1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A_fc2, a=math.sqrt(5))

        self._D = D
        self._H = H
        self._ks = ks
        self._r = r

    @property
    def lora_rank(self):
        return self._r

    def delta_conv(self):
        """Return ΔW for ``temporal_conv`` as a 3D tensor ``[D, D, ks]``."""
        return (self.A_conv @ self.B_conv).reshape(self._D, self._D, self._ks)

    def delta_fc1(self):
        return self.A_fc1 @ self.B_fc1

    def delta_fc2(self):
        return self.A_fc2 @ self.B_fc2


class LoRAExpertMLP(nn.Module):
    """LoRA wrapper around a (frozen) full-rank ``TemporalExpertMLP`` base.

    Holds the base expert plus an ``nn.ModuleList`` of ``LoRAAdapter``s,
    one per stage ``>= 1`` that has trained against this slot. Forward
    reconstructs each weight matrix as
    ``W_eff = W_base + Σ_{i < current_task_idx} A_i · B_i`` and runs the
    standard ``TemporalExpertMLP``-style pipeline using the base's biases,
    LayerNorm, and activation.

    ``current_task_idx`` semantics:
      * ``0`` -> only base, no adapters active (stage-0 evaluation).
      * ``s >= 1`` -> base + ``lora_adapters[0..s-1]`` (s adapters active).

    During training of stage ``t``, ``lora_adapters[t-1]`` is the trainable
    adapter; all earlier adapters and the base are frozen.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.lora_adapters = nn.ModuleList()
        self.current_task_idx = 0

    def append_adapter(self, lora_rank):
        """Append a new trainable ``LoRAAdapter`` for the upcoming stage."""
        D = self.base.fc1.in_features
        H = self.base.fc1.out_features
        ks = self.base.temporal_conv.kernel_size[0]
        device = next(self.base.parameters()).device
        adapter = LoRAAdapter(D, H, ks, lora_rank).to(device)
        self.lora_adapters.append(adapter)

    def freeze_active_adapter(self):
        """Mark the most recently-added adapter as frozen."""
        if len(self.lora_adapters) > 0:
            for p in self.lora_adapters[-1].parameters():
                p.requires_grad = False

    def _effective_weights(self):
        """Return ``(W_conv_eff, W_fc1_eff, W_fc2_eff)`` summing the active
        adapters into the base. Active count is
        ``min(current_task_idx, len(lora_adapters))``.
        """
        b = self.base
        n_active = max(0, min(int(self.current_task_idx), len(self.lora_adapters)))
        W_conv = b.temporal_conv.weight
        W_fc1 = b.fc1.weight
        W_fc2 = b.fc2.weight
        for i in range(n_active):
            adapter = self.lora_adapters[i]
            W_conv = W_conv + adapter.delta_conv()
            W_fc1 = W_fc1 + adapter.delta_fc1()
            W_fc2 = W_fc2 + adapter.delta_fc2()
        return W_conv, W_fc1, W_fc2

    def forward(self, x):
        """Mirror of ``TemporalExpertMLP.forward`` with effective weights
        substituted in. Biases, LayerNorm, activation, dropout all come
        from the base.
        """
        b = self.base
        W_conv, W_fc1, W_fc2 = self._effective_weights()

        x_t = x.transpose(1, 2)
        x_conv = F.conv1d(
            x_t, W_conv,
            bias=b.temporal_conv.bias,
            padding=b.temporal_conv.padding[0],
        )
        x_conv = x_conv.transpose(1, 2)
        x_conv = b.activation(x_conv)
        h = b.temporal_norm(x + x_conv)

        out = F.linear(h, W_fc1, b.fc1.bias)
        out = b.activation(out)
        out = b.dropout(out)
        out = F.linear(out, W_fc2, b.fc2.bias)
        return out


def convert_to_lora_after_stage_0(seq_moe):
    """Wrap each ``TemporalExpertMLP`` slot in a ``LoRAExpertMLP``. Freezes
    the base in place. No SVD truncation is applied -- the base remains
    full-rank, matching the LoRA-baseline design (stage 0 = pretraining,
    full-rank base, subsequent stages = LoRA adapters).
    Idempotent: slots already wrapped are skipped.
    """
    for i in range(len(seq_moe.experts)):
        e = seq_moe.experts[i]
        if isinstance(e, LoRAExpertMLP):
            continue
        if not isinstance(e, TemporalExpertMLP):
            continue
        for p in e.parameters():
            p.requires_grad = False
        device = next(e.parameters()).device
        wrapper = LoRAExpertMLP(e).to(device)
        seq_moe.experts[i] = wrapper


def append_lora_adapter(seq_moe, lora_rank):
    """For each ``LoRAExpertMLP`` slot, append a fresh trainable
    ``LoRAAdapter`` (zero-init B, kaiming-init A so initial delta is 0).
    """
    for i in range(len(seq_moe.experts)):
        slot = seq_moe.experts[i]
        if not isinstance(slot, LoRAExpertMLP):
            raise TypeError(
                f"slot {i} is {type(slot).__name__}, expected LoRAExpertMLP. "
                "Did you forget to call convert_to_lora_after_stage_0?"
            )
        slot.append_adapter(lora_rank)


def freeze_active_lora_adapter(seq_moe):
    """For each ``LoRAExpertMLP`` slot, freeze the most recently-added
    adapter (called at end-of-stage to lock the just-trained adapter)."""
    for i in range(len(seq_moe.experts)):
        slot = seq_moe.experts[i]
        if isinstance(slot, LoRAExpertMLP):
            slot.freeze_active_adapter()


# =============================================================================
# Encoder-layer reservation: extends ``ours`` (cl_method='ours') from MoE-only
# to MoE + encoder Linear / Conv1d(k=1) layers. Mirrors StackedExpertMLP at the
# per-layer granularity used by ``apply_lowrank_to_encoders`` in
# ``src/analysis/eval_lowrank_experts.py``.
# =============================================================================


class LowRankLinear(nn.Module):
    """Frozen rank-r factored drop-in replacement for ``nn.Linear``.

    Stores a 2D weight as ``(U, S, V_h)`` SVD factors plus optional bias.
    Forward reconstructs ``W = (U * S) @ V_h`` and applies ``F.linear``,
    so the output equals running an ``nn.Linear`` whose weight was
    rank-k SVD-truncated.
    """

    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = max(1, min(rank, in_features, out_features))
        self.U = nn.Parameter(torch.zeros(out_features, k), requires_grad=False)
        self.S = nn.Parameter(torch.zeros(k), requires_grad=False)
        self.Vh = nn.Parameter(torch.zeros(k, in_features), requires_grad=False)
        self._k = k
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def _weight(self):
        return _reconstruct(self.U, self.S, self.Vh)

    def forward(self, x):
        return F.linear(x, self._weight(), self.bias)

    @classmethod
    def from_linear(cls, linear, rank):
        new = cls(linear.in_features, linear.out_features, rank,
                  bias=(linear.bias is not None))
        with torch.no_grad():
            U, S, Vh = _svd_factors(linear.weight.detach(), rank)
            new.U.copy_(U)
            new.S.copy_(S)
            new.Vh.copy_(Vh)
            if linear.bias is not None:
                new.bias.copy_(linear.bias.detach())
        for p in new.parameters():
            p.requires_grad = False
        return new


class LowRankConv1dK1(nn.Module):
    """Frozen rank-r factored drop-in replacement for ``nn.Conv1d`` with
    ``kernel_size=1``. The (out, in, 1) weight reshapes to (out, in) for SVD;
    forward applies ``F.conv1d`` with the reconstructed (out, in, 1) kernel.
    """

    def __init__(self, in_channels, out_channels, rank, bias=True,
                 stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        if groups != 1:
            raise ValueError(
                f"LowRankConv1dK1 only supports groups=1, got {groups}.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        k = max(1, min(rank, in_channels, out_channels))
        self.U = nn.Parameter(torch.zeros(out_channels, k), requires_grad=False)
        self.S = nn.Parameter(torch.zeros(k), requires_grad=False)
        self.Vh = nn.Parameter(torch.zeros(k, in_channels), requires_grad=False)
        self._k = k
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def _weight(self):
        W2d = _reconstruct(self.U, self.S, self.Vh)
        return W2d.reshape(self.out_channels, self.in_channels, 1)

    def forward(self, x):
        return F.conv1d(x, self._weight(), self.bias,
                        stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)

    @classmethod
    def from_conv1d(cls, conv, rank):
        if tuple(conv.kernel_size) != (1,):
            raise ValueError(
                f"LowRankConv1dK1 expects kernel_size=(1,), got {conv.kernel_size}.")
        new = cls(
            conv.in_channels, conv.out_channels, rank,
            bias=(conv.bias is not None),
            stride=conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride,
            padding=conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding,
            dilation=conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation,
            groups=conv.groups,
        )
        with torch.no_grad():
            W = conv.weight.detach()
            out_c, in_c, ks = W.shape
            assert ks == 1
            U, S, Vh = _svd_factors(W.reshape(out_c, in_c), rank)
            new.U.copy_(U)
            new.S.copy_(S)
            new.Vh.copy_(Vh)
            if conv.bias is not None:
                new.bias.copy_(conv.bias.detach())
        for p in new.parameters():
            p.requires_grad = False
        return new


class _StackedLowRankBase(nn.Module):
    """Common forward / append / reserve logic shared by the Linear and
    Conv1d(k=1) stacked wrappers. Output = sum of components ``[0..idx]``
    where ``idx = current_task_idx`` clamped to ``len(components)-1``.

    Each component is either a frozen low-rank module (``LowRankLinear`` /
    ``LowRankConv1dK1``) for already-reserved stages, or a full-rank trainable
    module (``nn.Linear`` / ``nn.Conv1d``) for the current stage. Subclasses
    set ``_lowrank_cls`` and ``_active_cls`` accordingly.
    """

    def __init__(self):
        super().__init__()
        self.components = nn.ModuleList()
        self.current_task_idx = 0

    def append_active(self, fresh_layer):
        self.components.append(fresh_layer)

    def forward(self, x):
        if len(self.components) == 0:
            return torch.zeros_like(x)
        n_use = min(int(self.current_task_idx) + 1, len(self.components))
        if n_use <= 0:
            n_use = 1
        out = self.components[0](x)
        for i in range(1, n_use):
            out = out + self.components[i](x)
        return out


class StackedLowRankLinear(_StackedLowRankBase):
    """Stacked low-rank wrapper for an ``nn.Linear``. Drop-in replacement
    that exposes ``in_features`` / ``out_features`` for downstream code that
    introspects shapes.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

    def reserve_active(self, rank):
        """SVD-truncate the last (active) component to rank-r and replace it
        in place with a frozen ``LowRankLinear``. No-op if it is already low-rank.
        """
        if len(self.components) == 0:
            return
        last = self.components[-1]
        if isinstance(last, LowRankLinear):
            return
        if not isinstance(last, nn.Linear):
            raise TypeError(
                f"Expected nn.Linear active component, got {type(last).__name__}.")
        device = last.weight.device
        lr = LowRankLinear.from_linear(last, rank).to(device)
        self.components[-1] = lr

    @classmethod
    def from_linear(cls, linear, rank):
        """Build a Stacked module containing one frozen low-rank component
        derived from ``linear``. Use at end of the FIRST stage in which
        ``linear`` was trained.
        """
        new = cls(linear.in_features, linear.out_features,
                  bias=(linear.bias is not None))
        device = linear.weight.device
        new.components.append(LowRankLinear.from_linear(linear, rank).to(device))
        return new


class StackedLowRankConv1dK1(_StackedLowRankBase):
    """Stacked low-rank wrapper for ``nn.Conv1d`` with kernel_size=1."""

    def __init__(self, in_channels, out_channels, bias=True,
                 stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def reserve_active(self, rank):
        if len(self.components) == 0:
            return
        last = self.components[-1]
        if isinstance(last, LowRankConv1dK1):
            return
        if not isinstance(last, nn.Conv1d):
            raise TypeError(
                f"Expected nn.Conv1d active component, got {type(last).__name__}.")
        device = last.weight.device
        lr = LowRankConv1dK1.from_conv1d(last, rank).to(device)
        self.components[-1] = lr

    @classmethod
    def from_conv1d(cls, conv, rank):
        if tuple(conv.kernel_size) != (1,):
            raise ValueError(
                f"StackedLowRankConv1dK1 expects kernel_size=(1,), got {conv.kernel_size}.")
        new = cls(
            conv.in_channels, conv.out_channels,
            bias=(conv.bias is not None),
            stride=conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride,
            padding=conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding,
            dilation=conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation,
            groups=conv.groups,
        )
        device = conv.weight.device
        new.components.append(LowRankConv1dK1.from_conv1d(conv, rank).to(device))
        return new


def _make_fresh_linear_active(stack):
    """Build a fresh ``nn.Linear`` matching a ``StackedLowRankLinear``'s shape
    with weight + bias zero-initialised so the new component contributes
    zero at the start of the upcoming stage.
    """
    device = stack.components[0].U.device  # any frozen comp has params on device
    fresh = nn.Linear(stack.in_features, stack.out_features, bias=stack.has_bias)
    with torch.no_grad():
        fresh.weight.zero_()
        if fresh.bias is not None:
            fresh.bias.zero_()
    return fresh.to(device)


def _make_fresh_conv1dk1_active(stack):
    device = stack.components[0].U.device
    fresh = nn.Conv1d(
        stack.in_channels, stack.out_channels, kernel_size=1,
        stride=stack.stride, padding=stack.padding, dilation=stack.dilation,
        groups=stack.groups, bias=stack.has_bias,
    )
    with torch.no_grad():
        fresh.weight.zero_()
        if fresh.bias is not None:
            fresh.bias.zero_()
    return fresh.to(device)


def _is_target_layer(module):
    """Predicate: ``module`` is a layer that we (a) know how to SVD-reserve
    and (b) want to wrap when its weight is currently trainable. Mirrors
    ``apply_lowrank_to_encoders`` in eval_lowrank_experts.py.
    """
    if isinstance(module, nn.Linear):
        return True
    if isinstance(module, nn.Conv1d) and tuple(module.kernel_size) == (1,):
        return True
    return False


def _replace_submodule(parent, attr, new_module):
    """Replace an attribute on a parent ``nn.Module`` with ``new_module``.
    Handles ``nn.ModuleList`` / ``nn.ModuleDict`` accessors via ``[idx]``.
    """
    setattr(parent, attr, new_module)


def _walk_and_collect_layer_sites(encoder):
    """Yield ``(parent_module, attr_name, child_module)`` for every direct
    child that is a target layer (``nn.Linear`` / ``nn.Conv1d`` k=1) OR a
    ``StackedLowRank{Linear,Conv1dK1}`` wrapper. Recurses into submodules
    EXCEPT through StackedLowRank* wrappers -- the wrapper's components list
    is managed by the wrapper itself (``reserve_active`` /
    ``append_active``); a manual recursion into ``components`` would double-
    wrap the trainable active layer.
    """
    def _rec(parent):
        for attr, child in parent.named_children():
            is_stack = isinstance(child, _StackedLowRankBase)
            if _is_target_layer(child) or is_stack:
                yield parent, attr, child
            if is_stack:
                continue  # don't descend into the stack's internal components
            yield from _rec(child)
    yield from _rec(encoder)


def reserve_or_wrap_encoder_layer(parent, attr, child, rank):
    """Apply end-of-stage reservation for a single (parent, attr, child) site.

      * ``child`` is a plain ``nn.Linear``/``nn.Conv1d(k=1)`` whose weight is
        trainable -> wrap it in a ``StackedLowRank*`` containing one frozen
        low-rank component derived from ``child``'s SVD.
      * ``child`` is a ``StackedLowRank*`` whose last (active) component is
        full-rank -> SVD-truncate that component in place.
      * Otherwise (frozen plain layer or already-reserved active) -> no-op.
    """
    if isinstance(child, nn.Linear) and child.weight.requires_grad:
        new = StackedLowRankLinear.from_linear(child, rank).to(child.weight.device)
        _replace_submodule(parent, attr, new)
        return 'wrapped_linear'
    if (isinstance(child, nn.Conv1d) and tuple(child.kernel_size) == (1,)
            and child.weight.requires_grad):
        new = StackedLowRankConv1dK1.from_conv1d(child, rank).to(child.weight.device)
        _replace_submodule(parent, attr, new)
        return 'wrapped_conv1dk1'
    if isinstance(child, (StackedLowRankLinear, StackedLowRankConv1dK1)):
        if len(child.components) > 0 and not isinstance(
                child.components[-1], (LowRankLinear, LowRankConv1dK1)):
            child.reserve_active(rank)
            return 'reserved_active'
    return 'noop'


def reserve_encoder_layers(encoders, rank):
    """End-of-stage reservation pass over all encoders. For every trainable
    Linear / Conv1d(k=1) layer: wrap (first stage that trained it) or
    SVD-truncate the active component (subsequent stages). Encoders are
    deduplicated by ``id()`` to avoid double-wrapping shared encoders.

    Returns a counter dict: ``{'wrapped_linear': N, 'wrapped_conv1dk1': N,
    'reserved_active': N, 'noop': N}``.
    """
    counts = {'wrapped_linear': 0, 'wrapped_conv1dk1': 0,
              'reserved_active': 0, 'noop': 0}
    seen = set()
    for enc in encoders.values():
        if id(enc) in seen:
            continue
        seen.add(id(enc))
        # Materialize sites first because we mutate the tree as we go.
        sites = list(_walk_and_collect_layer_sites(enc))
        for parent, attr, child in sites:
            outcome = reserve_or_wrap_encoder_layer(parent, attr, child, rank)
            counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def append_fresh_encoder_active_components(encoders):
    """For each ``StackedLowRank*`` module inside any encoder, append a fresh
    zero-initialised trainable component (``nn.Linear`` / ``nn.Conv1d(k=1)``)
    matching the wrapper's shape. Forward output is unchanged at the moment
    of append because the new component is zeroed out.
    """
    counts = {'append_linear': 0, 'append_conv1dk1': 0}
    seen = set()
    for enc in encoders.values():
        if id(enc) in seen:
            continue
        seen.add(id(enc))
        for module in enc.modules():
            if isinstance(module, StackedLowRankLinear):
                module.append_active(_make_fresh_linear_active(module))
                counts['append_linear'] += 1
            elif isinstance(module, StackedLowRankConv1dK1):
                module.append_active(_make_fresh_conv1dk1_active(module))
                counts['append_conv1dk1'] += 1
    return counts


def set_current_task_idx_for_encoders(encoders, task_idx):
    """Propagate ``current_task_idx`` to every ``StackedLowRank*`` module
    in the given encoders dict. Mirrors :func:`set_current_task_idx` for
    in-model stacks.
    """
    seen = set()
    for enc in encoders.values():
        if id(enc) in seen:
            continue
        seen.add(id(enc))
        for module in enc.modules():
            if isinstance(module, (StackedLowRankLinear, StackedLowRankConv1dK1)):
                module.current_task_idx = int(task_idx)


def trainable_active_encoder_params(encoders):
    """Return the parameters of every ``StackedLowRank*`` module's currently-
    active (full-rank, last) component across all encoders. Useful for
    selectively unfreezing only the active components at the start of a
    new stage when ``--svd_target=moe_and_encoder``.
    """
    out = []
    seen = set()
    for enc in encoders.values():
        if id(enc) in seen:
            continue
        seen.add(id(enc))
        for module in enc.modules():
            if isinstance(module, (StackedLowRankLinear, StackedLowRankConv1dK1)):
                if len(module.components) == 0:
                    continue
                last = module.components[-1]
                if isinstance(last, (LowRankLinear, LowRankConv1dK1)):
                    continue
                out.extend(p for p in last.parameters() if p is not None)
    return out


# =============================================================================
# Encoder-layer LoRA: per-stage rank-r adapters on encoder nn.Linear and
# nn.Conv1d(kernel_size=1) layers. Mirrors LoRAExpertMLP at the per-layer
# granularity used by ``apply_lowrank_to_encoders`` in
# ``src/analysis/eval_lowrank_experts.py`` and the StackedLowRank* classes
# above. Triggered by --cl_method=lora --cl_target=moe_and_encoder.
# =============================================================================


class _LoRALayerAdapter(nn.Module):
    """A single rank-r delta ``A · B`` for a 2D weight matrix of shape
    ``[out, in]``. Standard LoRA init: ``A ~ kaiming_uniform_``, ``B = 0``
    so the initial delta is exactly zero (forward output is unchanged at
    the moment of append). Gradients flow to ``B`` first, then to ``A``
    once ``B`` becomes non-zero.
    """

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        r = max(1, int(rank))
        self.A = nn.Parameter(torch.empty(out_features, r))
        self.B = nn.Parameter(torch.zeros(r, in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self._r = r

    @property
    def lora_rank(self):
        return self._r

    def delta(self):
        return self.A @ self.B


class LoRALinearWrapper(nn.Module):
    """Drop-in replacement for ``nn.Linear`` whose effective weight is
    ``base.weight + Σ_{i < current_task_idx} adapters[i].delta()``. The
    base ``nn.Linear`` is frozen at end of stage 0 (when this wrapper is
    constructed). Each subsequent stage appends a fresh trainable
    ``_LoRALayerAdapter``; after that stage trains, the adapter is frozen
    via ``freeze_active_adapter``.

    ``current_task_idx`` semantics mirror :class:`LoRAExpertMLP`:
      * ``0`` -> base only (stage-0 inference).
      * ``s >= 1`` -> base + ``lora_adapters[0..s-1]`` (s adapters active).
    """

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.lora_adapters = nn.ModuleList()
        self.current_task_idx = 0

    def append_adapter(self, rank):
        device = self.base.weight.device
        adapter = _LoRALayerAdapter(self.in_features, self.out_features, rank).to(device)
        self.lora_adapters.append(adapter)

    def freeze_active_adapter(self):
        if len(self.lora_adapters) > 0:
            for p in self.lora_adapters[-1].parameters():
                p.requires_grad = False

    def _effective_weight(self):
        n_active = max(0, min(int(self.current_task_idx), len(self.lora_adapters)))
        W = self.base.weight
        for i in range(n_active):
            W = W + self.lora_adapters[i].delta()
        return W

    def forward(self, x):
        return F.linear(x, self._effective_weight(), self.base.bias)

    @classmethod
    def from_linear(cls, linear):
        """Wrap ``linear`` (will be the frozen stage-0 base). Caller is
        responsible for freezing the base's params.
        """
        new = cls(linear)
        for p in new.base.parameters():
            p.requires_grad = False
        return new


class LoRAConv1dK1Wrapper(nn.Module):
    """Drop-in replacement for ``nn.Conv1d`` with ``kernel_size=1``. Same
    pattern as :class:`LoRALinearWrapper`: frozen base + per-stage low-rank
    adapters summed into the effective ``[out, in, 1]`` kernel.
    """

    def __init__(self, base):
        super().__init__()
        if tuple(base.kernel_size) != (1,):
            raise ValueError(
                f"LoRAConv1dK1Wrapper expects kernel_size=(1,), got {base.kernel_size}.")
        self.base = base
        self.in_channels = base.in_channels
        self.out_channels = base.out_channels
        # Cache stride/padding/dilation/groups as ints for the conv1d call.
        self.stride = base.stride[0] if isinstance(base.stride, tuple) else base.stride
        self.padding = base.padding[0] if isinstance(base.padding, tuple) else base.padding
        self.dilation = base.dilation[0] if isinstance(base.dilation, tuple) else base.dilation
        self.groups = base.groups
        self.lora_adapters = nn.ModuleList()
        self.current_task_idx = 0

    def append_adapter(self, rank):
        device = self.base.weight.device
        adapter = _LoRALayerAdapter(self.in_channels, self.out_channels, rank).to(device)
        self.lora_adapters.append(adapter)

    def freeze_active_adapter(self):
        if len(self.lora_adapters) > 0:
            for p in self.lora_adapters[-1].parameters():
                p.requires_grad = False

    def _effective_weight(self):
        n_active = max(0, min(int(self.current_task_idx), len(self.lora_adapters)))
        W2d = self.base.weight.reshape(self.out_channels, self.in_channels)
        for i in range(n_active):
            W2d = W2d + self.lora_adapters[i].delta()
        return W2d.reshape(self.out_channels, self.in_channels, 1)

    def forward(self, x):
        return F.conv1d(x, self._effective_weight(), self.base.bias,
                        stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)

    @classmethod
    def from_conv1d(cls, conv):
        new = cls(conv)
        for p in new.base.parameters():
            p.requires_grad = False
        return new


def _walk_and_collect_lora_layer_sites(encoder):
    """Same as ``_walk_and_collect_layer_sites`` but recognizes
    ``LoRALinearWrapper`` / ``LoRAConv1dK1Wrapper`` as already-wrapped
    layers and prunes recursion through them so we don't double-wrap the
    inner ``base`` ``nn.Linear`` / ``nn.Conv1d``.
    """
    def _rec(parent):
        for attr, child in parent.named_children():
            is_wrap = isinstance(child, (LoRALinearWrapper, LoRAConv1dK1Wrapper))
            if _is_target_layer(child) or is_wrap:
                yield parent, attr, child
            if is_wrap:
                continue
            yield from _rec(child)
    yield from _rec(encoder)


def convert_encoder_layers_to_lora(encoders):
    """End of stage 0: wrap every trainable encoder ``nn.Linear`` /
    ``nn.Conv1d(k=1)`` in a LoRA wrapper with the just-trained weights as
    the frozen base. Idempotent for layers already wrapped.

    Mirrors :func:`convert_to_lora_after_stage_0` for MoE expert slots.
    Returns counts of wrapped layers per type.
    """
    counts = {'wrapped_linear': 0, 'wrapped_conv1dk1': 0, 'noop': 0}
    seen = set()
    for enc in encoders.values():
        if id(enc) in seen:
            continue
        seen.add(id(enc))
        sites = list(_walk_and_collect_lora_layer_sites(enc))
        for parent, attr, child in sites:
            if isinstance(child, nn.Linear) and child.weight.requires_grad:
                _replace_submodule(parent, attr,
                                   LoRALinearWrapper.from_linear(child).to(
                                       child.weight.device))
                counts['wrapped_linear'] += 1
            elif (isinstance(child, nn.Conv1d) and tuple(child.kernel_size) == (1,)
                  and child.weight.requires_grad):
                _replace_submodule(parent, attr,
                                   LoRAConv1dK1Wrapper.from_conv1d(child).to(
                                       child.weight.device))
                counts['wrapped_conv1dk1'] += 1
            else:
                counts['noop'] += 1
    return counts


def append_fresh_encoder_lora_adapters(encoders, rank):
    """Append a fresh trainable rank-r LoRA adapter to every
    ``LoRA{Linear,Conv1dK1}Wrapper`` in every encoder. Mirrors
    :func:`append_lora_adapter` for MoE expert slots.
    """
    counts = {'append_linear': 0, 'append_conv1dk1': 0}
    seen = set()
    for enc in encoders.values():
        if id(enc) in seen:
            continue
        seen.add(id(enc))
        for module in enc.modules():
            if isinstance(module, LoRALinearWrapper):
                module.append_adapter(rank)
                counts['append_linear'] += 1
            elif isinstance(module, LoRAConv1dK1Wrapper):
                module.append_adapter(rank)
                counts['append_conv1dk1'] += 1
    return counts


def freeze_active_encoder_lora_adapters(encoders):
    """Freeze the most recently-added LoRA adapter on every encoder LoRA
    wrapper. Called at end of every stage > 0 (paired with
    :func:`append_fresh_encoder_lora_adapters` for the next stage).
    """
    counts = {'frozen_linear': 0, 'frozen_conv1dk1': 0}
    seen = set()
    for enc in encoders.values():
        if id(enc) in seen:
            continue
        seen.add(id(enc))
        for module in enc.modules():
            if isinstance(module, LoRALinearWrapper):
                module.freeze_active_adapter()
                counts['frozen_linear'] += 1
            elif isinstance(module, LoRAConv1dK1Wrapper):
                module.freeze_active_adapter()
                counts['frozen_conv1dk1'] += 1
    return counts


def set_current_task_idx_for_encoder_lora(encoders, task_idx):
    """Propagate ``current_task_idx`` to every ``LoRA{Linear,Conv1dK1}Wrapper``
    in the given encoders. Mirrors :func:`set_current_task_idx_for_encoders`
    for the StackedLowRank* classes.
    """
    seen = set()
    for enc in encoders.values():
        if id(enc) in seen:
            continue
        seen.add(id(enc))
        for module in enc.modules():
            if isinstance(module, (LoRALinearWrapper, LoRAConv1dK1Wrapper)):
                module.current_task_idx = int(task_idx)
