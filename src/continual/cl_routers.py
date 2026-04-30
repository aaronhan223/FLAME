"""Continual-learning replacements for ``ModalityRouter``.

Two drop-in variants:

  - ``ColumnGrowModalityRouter``: subclass of ``ModalityRouter``. ``w_gate``
    and ``w_noise`` remain as ``nn.Parameter``s but grow over time as
    new experts are added. A backward hook zeroes gradients in the first
    ``_num_frozen_cols`` columns, so columns associated with already-reserved
    experts behave as frozen even though they are still ``nn.Parameter``s.

  - ``PerTaskModalityRouter``: holds an ``nn.ModuleList`` of
    ``ModalityRouter``s, one per task that has been trained. ``current_task_idx``
    selects which router fires at forward. Frozen prior-task routers have
    ``requires_grad=False`` on all their parameters. Output gates are
    zero-padded up to the current global expert pool size so downstream
    ``SparseDispatcher3D`` sees a fixed gate width.

Both variants expose the same ``forward(x, train, noise_epsilon)`` signature
as ``ModalityRouter`` so they slot into ``seq_moe.routers[modality]`` without
any change to ``SeqMoE.forward``.
"""

import torch
from torch import nn

from src.fusemoe_multitask.moe import ModalityRouter


class ColumnGrowModalityRouter(ModalityRouter):
    """``ModalityRouter`` whose ``w_gate``/``w_noise`` grow column-wise.

    The first ``_num_frozen_cols`` columns are kept "frozen" via a backward
    hook that zeroes out their gradients before the optimizer step. Columns
    can be appended via :py:meth:`grow` and the freeze boundary advanced via
    :py:meth:`freeze_first`.
    """

    def __init__(self, embed_dim, num_experts, top_k=4, gating='softmax',
                 noisy_gating=True):
        super().__init__(embed_dim, num_experts, top_k=top_k, gating=gating,
                         noisy_gating=noisy_gating)
        self._num_frozen_cols = 0
        self._gate_hook = self.w_gate.register_hook(self._gate_grad_filter)
        self._noise_hook = self.w_noise.register_hook(self._noise_grad_filter)

    def _gate_grad_filter(self, grad):
        if self._num_frozen_cols > 0:
            grad = grad.clone()
            grad[:, :self._num_frozen_cols] = 0.0
        return grad

    def _noise_grad_filter(self, grad):
        if self._num_frozen_cols > 0:
            grad = grad.clone()
            grad[:, :self._num_frozen_cols] = 0.0
        return grad

    def grow(self, num_new_experts):
        """Append ``num_new_experts`` zero-init columns to ``w_gate``/``w_noise``."""
        if num_new_experts <= 0:
            return
        cur = self.num_experts
        new_total = cur + num_new_experts
        device = self.w_gate.device
        dtype = self.w_gate.dtype

        if hasattr(self, '_gate_hook'):
            self._gate_hook.remove()
            self._noise_hook.remove()

        new_gate = torch.zeros(self.embed_dim, new_total, device=device, dtype=dtype)
        new_noise = torch.zeros(self.embed_dim, new_total, device=device, dtype=dtype)
        with torch.no_grad():
            new_gate[:, :cur].copy_(self.w_gate.data)
            new_noise[:, :cur].copy_(self.w_noise.data)

        # Replace the Parameter instances. ``register_parameter`` re-registers
        # under the same name so external lookups via ``self.w_gate`` keep working.
        self.register_parameter('w_gate', nn.Parameter(new_gate))
        self.register_parameter('w_noise', nn.Parameter(new_noise))
        self.num_experts = new_total

        self._gate_hook = self.w_gate.register_hook(self._gate_grad_filter)
        self._noise_hook = self.w_noise.register_hook(self._noise_grad_filter)

    def freeze_first(self, n_frozen_cols):
        """Set the freeze boundary so columns ``[0:n]`` receive zero gradient."""
        self._num_frozen_cols = int(n_frozen_cols)


class PerTaskModalityRouter(nn.Module):
    """List of per-task ``ModalityRouter``s with combined ("union of router
    heads") gating across routers ``[0..current_task_idx]``.

    For a sample passing through this wrapper at index ``t = current_task_idx``:

      1. Each router ``r_i`` for ``i in 0..t`` computes its clean logits
         over its own visible expert range ``[0, (i+1)*E)``.
      2. Logits are placed into a global ``[bs, num_experts]`` matrix. Slots
         not seen by router ``r_i`` are sentinel ``-inf``. Per-expert
         contributions across routers are reduced with the configured
         ``combine`` operator (``'mean'``, ``'sum'``, or ``'max'``).
      3. Noise (during training, when ``noisy_gating=True``) is taken from
         the latest active router only. Frozen routers contribute their
         (frozen) routing prior but no exploration noise.
      4. Top-K + softmax/laplace gate normalization runs once on the
         combined logits, producing a sparse ``[bs, num_experts]`` gate
         matrix that the dispatcher consumes unchanged.

    This matches design (B): "previous tasks use frozen routers, new tasks
    use the new router heads only" -- when ``set_current_task_idx(idx)`` is
    called for evaluation of prior task ``s``, only routers ``[0..s]``
    contribute, so experts beyond ``(s+1)*E`` get sentinel ``-inf`` and are
    structurally excluded from top-K.
    """

    def __init__(self, embed_dim, num_experts, top_k=4, gating='softmax',
                 noisy_gating=True, combine='mean'):
        super().__init__()
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.gating = gating
        self.noisy_gating = noisy_gating
        if combine not in ('mean', 'sum', 'max'):
            raise ValueError(f"combine must be 'mean'|'sum'|'max', got {combine!r}")
        self.combine = combine
        self.task_routers = nn.ModuleList([
            ModalityRouter(embed_dim, num_experts, top_k=top_k,
                           gating=gating, noisy_gating=noisy_gating)
        ])
        self.softmax = nn.Softmax(dim=1)
        self.num_experts = num_experts          # global current total
        self.current_task_idx = 0               # most recent active router idx

    @property
    def active_router(self):
        return self.task_routers[-1]

    def add_task_router(self, num_experts_total):
        """Freeze the currently-active router and append a new active one
        sized for ``num_experts_total`` (the new global pool size). The new
        router has full visibility: it can route to all frozen + new experts.
        """
        for p in self.task_routers[-1].parameters():
            p.requires_grad = False
        device = next(self.task_routers[-1].parameters()).device
        new_router = ModalityRouter(
            self.embed_dim, num_experts_total, top_k=self.top_k,
            gating=self.gating, noisy_gating=self.noisy_gating,
        ).to(device)
        self.task_routers.append(new_router)
        self.num_experts = num_experts_total
        self.current_task_idx = len(self.task_routers) - 1

    def freeze_active(self):
        """Mark the currently-active router as frozen (used between
        SVD-reserve and the next ``add_task_router`` call)."""
        for p in self.task_routers[-1].parameters():
            p.requires_grad = False

    def _router_clean_logits(self, router, x):
        """Compute clean logits for a single ``ModalityRouter`` over its
        visible expert range. Returns ``[bs, router.num_experts]``."""
        gate_input = router.temporal_pool(x)
        if router.gating == 'softmax':
            return gate_input @ router.w_gate, gate_input
        if router.gating == 'laplace':
            return -torch.cdist(gate_input, router.w_gate.t()), gate_input
        if router.gating == 'gaussian':
            return -torch.pow(torch.cdist(gate_input, router.w_gate.t()), 2), gate_input
        raise ValueError(f"Unknown gating {router.gating!r}")

    def forward(self, x, train=True, noise_epsilon=1e-2):
        idx = max(0, min(int(self.current_task_idx), len(self.task_routers) - 1))
        bs = x.shape[0]
        device = x.device
        dtype = x.dtype
        N = self.num_experts

        # Stack each contributing router's clean logits into [num_routers, bs, N]
        # with -inf in slots the router cannot see.
        contribs = torch.full((idx + 1, bs, N), float('-inf'), device=device, dtype=dtype)
        latest_gate_input = None
        for i in range(idx + 1):
            r = self.task_routers[i]
            n_i = r.num_experts
            L_i, gi = self._router_clean_logits(r, x)
            contribs[i, :, :n_i] = L_i
            if i == idx:
                latest_gate_input = gi

        # Combine across routers per expert. ``visibility`` is the number of
        # routers that "see" each expert; experts unseen by every contributing
        # router stay at ``-inf`` and are excluded from top-K.
        visibility = (~torch.isinf(contribs[:, 0, :])).sum(dim=0).to(dtype)
        finite_mask = ~torch.isinf(contribs)
        if self.combine == 'sum':
            zeroed = torch.where(finite_mask, contribs, torch.zeros_like(contribs))
            combined = zeroed.sum(dim=0)
        elif self.combine == 'mean':
            zeroed = torch.where(finite_mask, contribs, torch.zeros_like(contribs))
            combined = zeroed.sum(dim=0) / visibility.clamp(min=1).unsqueeze(0)
        else:  # 'max'
            combined = contribs.max(dim=0).values

        unseen = visibility == 0
        combined = combined.masked_fill(unseen.unsqueeze(0), float('-inf'))

        clean_logits = combined
        # Noise from the latest active router only -- exploration belongs to
        # the trainable head; frozen prior heads contribute a fixed routing
        # prior with no stochasticity.
        latest = self.task_routers[idx]
        if latest.noisy_gating and train:
            raw_noise = latest_gate_input @ latest.w_noise
            noise_stddev = latest.softplus(raw_noise) + noise_epsilon
            noise_full = torch.zeros((bs, N), device=device, dtype=dtype)
            noise_full[:, :latest.num_experts] = noise_stddev
            noise_full = noise_full.masked_fill(unseen.unsqueeze(0), 0.0)
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_full
            logits = noisy_logits
        else:
            logits = clean_logits
            noisy_logits = clean_logits
            noise_stddev = None

        # Top-K + softmax/laplace gate normalization on the combined logits.
        visible_count = int(visibility.gt(0).sum().item())
        k_pool = min(self.top_k + 1, max(1, visible_count))
        top_logits, top_indices = logits.topk(k_pool, dim=1)
        eff_k = min(self.top_k, max(1, visible_count))
        top_k_logits = top_logits[:, :eff_k]
        top_k_indices = top_indices[:, :eff_k]

        if self.gating == 'softmax':
            top_k_gates = self.softmax(top_k_logits)
        else:  # laplace, gaussian
            top_k_gates = torch.exp(
                top_k_logits
                - torch.logsumexp(top_k_logits, dim=1, keepdim=True)
            )

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # Simplified load: count of samples that selected each expert. The
        # noisy-prob load formula in ModalityRouter assumes a single router;
        # generalizing it across combined routers is messy and is unnecessary
        # for the balance-loss signal that drives the active head.
        load = (gates > 0).sum(dim=0).to(dtype)

        return gates, load
