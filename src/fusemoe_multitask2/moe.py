"""
@Shravan summary of updates

Core changes from the prior implementation: routing decisions are
made at the token level — each (batch, time-step) position is independently
routed to its top-k experts. Previously routing is based on sample-level. Ideally,
Gradient signal proportional to (batch × seq_len) rather than batch alone,
so experts see sufficient gradient to specialize from the first epoch.
And every token gets a gradient path through its assigned experts.

Also added new _load_balance_loss adapted from Switch Transformer:
the old CV loss is none differentiable to non-selected experts, the 
new loss provides gradient signal for all experts.

Added diversity loss to control the amount of cross-modal sharing, but
can be removed if needed.

Public interface is identical to the previous SeqMoE:
    forward(x_list, modality_labels, train, loss_coef) → (output_list, loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEConfig:
    """Configuration for the per-modality routed MoE.

    Args:
        num_experts: Number of expert networks in the shared pool.
        embed_dim: Per-modality embedding dimension (all modalities are
            projected to this dim before entering the MoE).
        moe_hidden_size: Hidden dimension of expert MLPs.
        modality_types: List of modality type names that will be routed,
            e.g., ['ts', 'txt', 'cxr', 'ecg']. One router is created per
            type. This should be the UNION of modalities across all tasks.
        top_k: Number of experts each token is routed to.
        temporal_kernel: Unused (kept for backward-compat config loading).
        noisy_gating: Whether to use noisy top-k gating for exploration.
        gating: Gating function type ('softmax' only; other values accepted
            for compat but ignored — token-level router always uses softmax).
        dropout: Dropout rate for expert MLPs.
        hidden_act: Unused (experts use ReLU; kept for backward-compat).
        use_default_router: If True, create a fallback router for any
            modality not in modality_types.
        diversity_coef: Coefficient for intra-task diversity loss.
        diversity_mode: 'spread' or 'concentrate'.
    """

    def __init__(
        self,
        num_experts,
        embed_dim,
        moe_hidden_size,
        modality_types,
        top_k=4,
        temporal_kernel=3,
        noisy_gating=True,
        gating='softmax',
        dropout=0.1,
        hidden_act="gelu",
        use_default_router=True,
        diversity_coef=0.1,
        diversity_mode='spread',
    ):
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.moe_hidden_size = moe_hidden_size
        self.modality_types = modality_types
        self.top_k = top_k
        self.temporal_kernel = temporal_kernel
        self.noisy_gating = noisy_gating
        self.gating = gating
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.use_default_router = use_default_router
        self.diversity_coef = diversity_coef
        self.diversity_mode = diversity_mode


class TemporalAttentionPool(nn.Module):
    """Learnable attention pooling over the temporal dimension.

    Produces a fixed-size sample representation from a variable-length
    sequence via scaled dot-product attention with a learned query.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, D].
        Returns:
            [batch_size, D] pooled representation.
        """
        scores = torch.matmul(x, self.query) * self.scale   # [bs, seq_len]
        attn_weights = torch.softmax(scores, dim=1)          # [bs, seq_len]
        return torch.einsum('bt,btd->bd', attn_weights, x)  # [bs, D]


class FeedForwardExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)


class ModalityRouter(nn.Module):
    """Token-level router for a single modality type.

    Each (batch, time-step) position independently selects its top-k experts
    from the shared pool, giving a routing gradient proportional to
    (batch × seq_len) rather than batch size alone.
    """

    def __init__(self, embed_dim, num_experts, top_k=4,
                 gating='softmax', noisy_gating=True, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating

        # Linear gate: input token → expert logits
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        # Noise head for exploration (Noisy Top-k Gating, Shazeer et al.)
        self.w_noise = nn.Linear(embed_dim, num_experts, bias=False)
        self.temporal_pool = TemporalAttentionPool(embed_dim)

    def forward(self, x_flat, train=True, noise_epsilon=1e-2):
        """Route a flat batch of tokens.

        Args:
            x_flat: [num_tokens, embed_dim].
            train: Whether to add exploration noise.

        Returns:
            top_k_probs:   [num_tokens, top_k]   — normalized gate weights.
            top_k_indices: [num_tokens, top_k]   — selected expert indices.
            all_logits:    [num_tokens, num_experts] — raw logits (for load loss).
        """
        logits = self.gate(x_flat)                             # [T, E]
        if self.noisy_gating and train:
            noise_std = F.softplus(self.w_noise(x_flat)) + noise_epsilon
            logits = logits + torch.randn_like(logits) * noise_std

        top_k_logits, top_k_indices = logits.topk(
            min(self.top_k, self.num_experts), dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        return top_k_probs, top_k_indices, logits


class SeqMoE(nn.Module):
    """Per-Modality Routed MoE for Multi-Task Multimodal Fusion.

    Token-level routing: each (batch, time-step) position independently
    selects top-k experts from the shared pool.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.embed_dim = config.embed_dim
        self.diversity_coef = config.diversity_coef
        self.diversity_mode = config.diversity_mode

        # One token-level router per modality type
        self.routers = nn.ModuleDict({
            mod: ModalityRouter(
                embed_dim=config.embed_dim,
                num_experts=config.num_experts,
                top_k=config.top_k,
                noisy_gating=config.noisy_gating,
            )
            for mod in config.modality_types
        })

        if config.use_default_router:
            self.default_router = ModalityRouter(
                embed_dim=config.embed_dim,
                num_experts=config.num_experts,
                top_k=config.top_k,
                noisy_gating=config.noisy_gating,
            )
        else:
            self.default_router = None

        # Shared expert pool
        self.experts = nn.ModuleList([
            FeedForwardExpert(config.embed_dim, config.moe_hidden_size, config.dropout)
            for _ in range(config.num_experts)
        ])

        self.output_norm = nn.LayerNorm(config.embed_dim)

    def _get_router(self, modality_label):
        if modality_label in self.routers:
            return self.routers[modality_label]
        elif self.default_router is not None:
            return self.default_router
        else:
            raise ValueError(
                f"Unknown modality '{modality_label}' and no default router. "
                f"Known types: {list(self.routers.keys())}"
            )

    @staticmethod
    def _load_balance_loss(all_logits, top_k_indices, num_experts):
        """
        Encourages uniform token distribution across experts.
        Loss = num_experts * sum_e(f_e * P_e) where:
            f_e = fraction of tokens dispatched to expert e (from hard top-k)
            P_e = mean softmax probability for expert e (differentiable)
        """
        probs = F.softmax(all_logits, dim=-1)           # [T, E]
        T = all_logits.shape[0]
        one_hot = torch.zeros(
            T, num_experts,
            device=all_logits.device, dtype=probs.dtype,
        )
        one_hot.scatter_(-1, top_k_indices, 1.0)        # [T, E]
        f = one_hot.mean(0)                             # [E]
        P = probs.mean(0)                               # [E]
        return num_experts * (f * P).sum()

    @staticmethod
    def _cv_squared(x):
        """Squared coefficient of variation (kept from original implementation)."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _diversity_loss(self, mean_probs_list):
        """ Goal: encourages different modalities to route to different experts.
        But I believe this loss could conflict with the balance loss.
        Just add them here since I mentioned in previous discussion.
        Pls feel free to remove it or keep it depend on the actual performance.

        Args:
            mean_probs_list: list of [num_experts] mean routing probability
                vectors, one per active modality in the current task.
        Returns:
            Scalar diversity loss.
        """
        if len(mean_probs_list) < 2:
            return torch.tensor(0.0, device=mean_probs_list[0].device)

        stacked = torch.stack(mean_probs_list)          # [M, E]
        normed = F.normalize(stacked, dim=1)
        sim = normed @ normed.t()                       # [M, M]
        n = len(mean_probs_list)
        mask = torch.triu(
            torch.ones(n, n, device=sim.device, dtype=torch.bool), diagonal=1)
        pairwise = sim[mask]

        if self.diversity_mode == 'spread':
            return pairwise.mean()
        else:
            return -pairwise.mean()

    def _dispatch(self, x_flat, router, train):
        """Token-level dispatch through the shared expert pool.

        Each token is sent to its top-k experts; weighted outputs are
        accumulated back with index_add_.

        Args:
            x_flat: [num_tokens, embed_dim] — flattened (batch × seq_len) tokens.
            router: ModalityRouter instance.
            train: Noisy-gating flag.

        Returns:
            output_flat: [num_tokens, embed_dim] — MoE output per token.
            balance_loss: Scalar load-balancing loss.
            mean_probs:   [num_experts] — mean routing probability (for diversity).
        """
        num_tokens = x_flat.shape[0]
        top_k_probs, top_k_indices, all_logits = router(x_flat, train=train)
        # top_k_probs:   [T, k]
        # top_k_indices: [T, k]

        output_flat = torch.zeros_like(x_flat)

        # Flatten top-k assignments to 1-D index arrays for masked dispatch
        flat_expert_idx = top_k_indices.view(-1)          # [T * k]
        flat_probs      = top_k_probs.view(-1)            # [T * k]
        flat_token_idx  = torch.arange(
            num_tokens, device=x_flat.device
        ).repeat_interleave(self.config.top_k)            # [T * k]

        for expert_i in range(self.num_experts):
            mask = flat_expert_idx == expert_i
            if not mask.any():
                continue
            tok_idx  = flat_token_idx[mask]               # token indices for expert_i
            probs_i  = flat_probs[mask].unsqueeze(-1)     # [n, 1]
            out_i    = self.experts[expert_i](x_flat[tok_idx])   # [n, D]
            output_flat.index_add_(0, tok_idx, out_i * probs_i)

        balance   = self._load_balance_loss(all_logits, top_k_indices, self.num_experts)
        mean_probs = F.softmax(all_logits, dim=-1).mean(0)   # [E]

        return output_flat, balance, mean_probs

    def forward(self, x_list, modality_labels, train=True, loss_coef=1e-2):
        """Process a task's multimodal inputs through per-modality token routing.

        Args:
            x_list: List of [seq_len, batch_size, embed_dim] tensors (seq-first).
                    One tensor per active modality. Modalities may have different
                    seq_len values.
            modality_labels: List of modality type names parallel to x_list,
                    e.g. ['ts', 'txt'] or ['ts', 'cxr', 'txt'].
            train: Whether in training mode (enables noisy gating).
            loss_coef: Balance loss coefficient.

        Returns:
            output_list: List of [seq_len, bs, embed_dim] in the same order
                         as x_list.
            total_loss: Scalar (balance loss + diversity loss).
        """
        assert len(x_list) == len(modality_labels), (
            f"Got {len(x_list)} inputs but {len(modality_labels)} labels"
        )

        output_list    = []
        mean_probs_list = []
        balance_loss   = torch.tensor(0.0, device=x_list[0].device)

        for x, mod_label in zip(x_list, modality_labels):
            if torch.isnan(x).any():
                return None, None

            seq_len, bs, D = x.shape
            # Flatten to token stream: [bs, seq_len, D] → [bs*seq_len, D]
            x_flat = x.transpose(0, 1).reshape(-1, D)

            router = self._get_router(mod_label)
            output_flat, bal, mean_probs = self._dispatch(x_flat, router, train)

            # Reshape → normalize → restore seq-first
            output = output_flat.view(bs, seq_len, D)   # [bs, seq_len, D]
            output = self.output_norm(output)
            output = output.transpose(0, 1)             # [seq_len, bs, D]

            output_list.append(output)
            mean_probs_list.append(mean_probs)
            balance_loss = balance_loss + bal

        balance_loss  = balance_loss * loss_coef
        diversity_loss = self._diversity_loss(mean_probs_list) * self.diversity_coef
        total_loss    = balance_loss + diversity_loss

        return output_list, total_loss
