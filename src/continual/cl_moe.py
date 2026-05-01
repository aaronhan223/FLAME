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
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    k = min(rank, len(S))
    return U[:, :k].to(W.dtype), S[:k].to(W.dtype), Vh[:k, :].to(W.dtype)


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
    """Set ``current_task_idx`` on every ``PerTaskModalityRouter`` and every
    ``StackedExpertMLP`` in the model. The router uses it to pick which
    per-task head fires; the stacked expert uses it to slice the components
    list (sum components ``[0..task_idx]``). No-op for plain modules."""
    for module in model.modules():
        if isinstance(module, PerTaskModalityRouter):
            module.current_task_idx = int(task_idx)
        elif isinstance(module, StackedExpertMLP):
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
