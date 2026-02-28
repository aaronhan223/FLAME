"""
@Shravan: Possible usage example:

Config:
    config = MoEConfig(
        num_experts=8,
        embed_dim=128,
        moe_hidden_size=256,
        modality_types=['ts', 'txt', 'cxr', 'ecg'],  # union across tasks
        top_k=2,
    )
    moe = SeqMoE(config)

Forward (called per task, with that task's active modalities):
    # T1 has TS + CXR + Text
    output_list, loss = moe(
        x_list=[proj_x_ts, proj_x_cxr, proj_x_txt],
        modality_labels=['ts', 'cxr', 'txt'],
    )

    # T2 has TS + Text
    output_list, loss = moe(
        x_list=[proj_x_ts, proj_x_txt],
        modality_labels=['ts', 'txt'],
    )

@Shravan: In TransformerCrossEncoderLayer (replaces old flatten/MoE/unflatten block):
    moe_output, balance_loss = self.moe(x_list, modality_labels=modality)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from src.activations import ACT2FN

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
        top_k: Number of experts each modality sequence is routed to.
        temporal_kernel: Kernel size for temporal convolution in experts.
        noisy_gating: Whether to use noisy top-k gating for exploration.
        gating: Gating function type ('softmax', 'laplace', 'gaussian').
        dropout: Dropout rate for expert MLPs.
        hidden_act: Activation function name for expert MLPs.
        use_default_router: If True, create a fallback router for any
            modality not in modality_types (useful for rare modalities).
        diversity_coef: Coefficient for intra-task diversity loss.
        diversity_mode: 'spread' (penalize routing overlap between
            modalities within a task) or 'concentrate' (encourage overlap).
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
    Output size is always embed_dim, regardless of seq_len.
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
        pooled = torch.einsum('bt,btd->bd', attn_weights, x) # [bs, D]
        return pooled

class SparseDispatcher3D:
    """Sparse dispatcher for sample-level routing with 3D tensors.

    Routes at the sample level (dim 0) and preserves the full temporal
    sequence (dim 1) for each sample. Each expert receives complete
    sequences [expert_bs, seq_len, D], not individual tokens.

    Operates in linear space (no log/exp conversion).
    """

    def __init__(self, num_experts, gates):
        """
        Args:
            num_experts: Number of experts.
            gates: [batch_size, num_experts] sparse gate weights.
        """
        self._gates = gates
        self._num_experts = num_experts

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = (
            torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        )
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """
        Args:
            inp: [batch_size, seq_len, D].
        Returns:
            List of num_experts tensors, each [expert_batch_i, seq_len, D].
        """
        inp_exp = inp[self._batch_index]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """
        Args:
            expert_out: List of [expert_batch_i, seq_len, D_out] tensors.
        Returns:
            [batch_size, seq_len, D_out].
        """
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            # [total, 1] → [total, 1, 1] to broadcast across seq & feat dims
            stitched = stitched * self._nonzero_gates.unsqueeze(-1)
        zeros = torch.zeros(
            self._gates.size(0), *stitched.shape[1:],
            requires_grad=True, device=stitched.device,
        )
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

class TemporalExpertMLP(nn.Module):
    """Expert network that processes a single-modality 3D sequence.

    Two stages:
      1. Temporal conv (1D along time): local time-step interactions with
         residual + LayerNorm. Weights are [D, D, kernel_size].
      2. Per-position MLP: feature transformation shared across time.
         Weights are [D, hidden] and [hidden, D].

    Input/output size is embed_dim (single modality), so the same expert
    can process TS sequences, Text sequences, ECG sequences, etc.
    """

    def __init__(self, input_size, hidden_size, temporal_kernel=3,
                 dropout=0.1, hidden_act="gelu"):
        super().__init__()
        self.temporal_conv = nn.Conv1d(
            input_size, input_size,
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
        )
        self.temporal_norm = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = ACT2FN[hidden_act]

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, D].
        Returns:
            [batch, seq_len, D].
        """
        # Temporal mixing
        x_conv = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        x_conv = self.activation(x_conv)
        h = self.temporal_norm(x + x_conv)

        out = self.fc1(h)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class ModalityRouter(nn.Module):
    """Router for a single modality type.

    Each modality type (TS, Text, CXR, ECG, ...) gets its own router.
    The router:
      1. Pools a variable-length modality sequence into a fixed-size
         representation via TemporalAttentionPool.
      2. Produces sparse top-k gate weights over the shared expert pool.

    Because the same router handles the same modality from ALL tasks
    (e.g., TS from T1 and TS from T2 both go through router_ts),
    inter-task sharing is structural: similar TS patterns from different
    tasks will route to similar experts through the shared gate weights.

    Args:
        embed_dim: Per-modality embedding dimension.
        num_experts: Size of the shared expert pool.
        top_k: Number of experts to select per sample.
        gating: Gating function type.
        noisy_gating: Whether to add exploration noise.
    """

    def __init__(self, embed_dim, num_experts, top_k=4,
                 gating='softmax', noisy_gating=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gating = gating
        self.noisy_gating = noisy_gating

        self.temporal_pool = TemporalAttentionPool(embed_dim)
        self.w_gate = nn.Parameter(torch.zeros(embed_dim, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(embed_dim, num_experts))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def forward(self, x, train=True, noise_epsilon=1e-2):
        """Route a single-modality 3D sequence.

        Args:
            x: [batch_size, seq_len, embed_dim] — one modality's sequence.
            train: Whether to add gating noise.

        Returns:
            gates: [batch_size, num_experts] sparse gate weights.
            load: [num_experts] estimated expert load.
        """
        gate_input = self.temporal_pool(x)  # [bs, embed_dim]

        if self.gating == 'softmax':
            clean_logits = gate_input @ self.w_gate
        elif self.gating == 'laplace':
            clean_logits = -torch.cdist(gate_input, self.w_gate.t())
        elif self.gating == 'gaussian':
            clean_logits = -torch.pow(
                torch.cdist(gate_input, self.w_gate.t()), 2)

        if self.noisy_gating and train:
            raw_noise_stddev = gate_input @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
            noisy_logits = clean_logits
            noise_stddev = torch.zeros_like(clean_logits)

        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]

        if self.gating == 'softmax':
            top_k_gates = self.softmax(top_k_logits)
        elif self.gating in ('laplace', 'gaussian'):
            top_k_gates = torch.exp(
                top_k_logits
                - torch.logsumexp(top_k_logits, dim=1, keepdim=True))

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.top_k < self.num_experts:
            load = self._prob_in_top_k(
                clean_logits, noisy_logits, noise_stddev, top_logits
            ).sum(0)
        else:
            load = (gates > 0).sum(0)

        return gates, load

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev,
                       noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.top_k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf(
            (clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf(
            (clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

class SeqMoE(nn.Module):
    """Per-Modality Routed MoE for Multi-Task Multimodal Fusion.

    Each modality type gets its own ModalityRouter. All routers send to
    a SHARED pool of TemporalExpertMLPs. This enables:

      - Cross-task sharing: TS from T1 and TS from T2 use the same router
        and experts, so shared temporal patterns transfer automatically.

      - Flexible modality combinations: each forward call specifies which
        modalities are active (via modality_labels). T1 can have 3
        modalities, T2 can have 2, etc.

      - Modality-specific routing: Text sequences develop text-specific
        routing patterns, TS sequences develop TS-specific patterns,
        while the shared expert pool captures cross-modal commonalities.

    Regularization:
      - Balance loss: per-router CV-squared on importance and load.
      - Diversity loss: controls routing overlap between modalities
        within the same task (spread or concentrate).
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.embed_dim = config.embed_dim
        self.diversity_coef = config.diversity_coef
        self.diversity_mode = config.diversity_mode

        # --- One router per modality type ---
        self.routers = nn.ModuleDict({
            mod: ModalityRouter(
                embed_dim=config.embed_dim,
                num_experts=config.num_experts,
                top_k=config.top_k,
                gating=config.gating,
                noisy_gating=config.noisy_gating,
            )
            for mod in config.modality_types
        })

        if config.use_default_router:
            self.default_router = ModalityRouter(
                embed_dim=config.embed_dim,
                num_experts=config.num_experts,
                top_k=config.top_k,
                gating=config.gating,
                noisy_gating=config.noisy_gating,
            )
        else:
            self.default_router = None

        #Shared expert pool
        self.experts = nn.ModuleList([
            TemporalExpertMLP(
                input_size=config.embed_dim,
                hidden_size=config.moe_hidden_size,
                temporal_kernel=config.temporal_kernel,
                dropout=config.dropout,
                hidden_act=config.hidden_act,
            )
            for _ in range(config.num_experts)
        ])

    def _get_router(self, modality_label):
        """Look up the router for a modality type, with fallback."""
        if modality_label in self.routers:
            return self.routers[modality_label]
        elif self.default_router is not None:
            return self.default_router
        else:
            raise ValueError(
                f"Unknown modality '{modality_label}' and no default "
                f"router configured. Known types: "
                f"{list(self.routers.keys())}"
            )

    @staticmethod
    def _cv_squared(x):
        """Squared coefficient of variation for load balancing."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _diversity_loss(self, all_gates):
        """Intra-task diversity loss on routing distributions.

        Computes pairwise cosine similarity between the average routing
        distributions of the modalities active in the current forward
        call (= current task).

        Args:
            all_gates: list of [bs, num_experts] gate tensors, one per
                       active modality in this task.
        Returns:
            Scalar loss. Positive = penalize overlap (spread mode),
            negative = encourage overlap (concentrate mode).
        """
        if len(all_gates) < 2:
            return torch.tensor(0.0, device=all_gates[0].device)

        # Average routing distribution per modality: [num_mod, num_experts]
        avg_dists = torch.stack([g.mean(dim=0) for g in all_gates])
        avg_dists_norm = F.normalize(avg_dists, dim=1)

        # Pairwise cosine similarity (upper triangle, no self-pairs)
        sim_matrix = avg_dists_norm @ avg_dists_norm.t()
        n = len(all_gates)
        mask = torch.triu(
            torch.ones(n, n, device=sim_matrix.device, dtype=torch.bool),
            diagonal=1,
        )
        pairwise_sim = sim_matrix[mask]

        if self.diversity_mode == 'spread':
            # High similarity → high loss → push modalities apart
            return pairwise_sim.mean()
        else:  # concentrate
            # Low similarity → high loss → pull modalities together
            return -pairwise_sim.mean()

    def forward(self, x_list, modality_labels, train=True, loss_coef=1e-2):
        """Process a task's multimodal inputs through per-modality routing.

        Each modality sequence is independently routed by its modality-
        specific router to the shared expert pool. Different tasks can
        pass different subsets of modalities.

        Args:
            x_list: List of tensors, each [seq_len, batch_size, embed_dim].
                    One tensor per active modality in the current task.
                    Modalities can have DIFFERENT seq_lens.
            modality_labels: List of modality type names corresponding
                    to x_list, e.g., ['ts', 'txt'] or ['ts', 'cxr', 'txt'].
                    Used to select the appropriate router for each input.
            train: Whether in training mode (enables noisy gating).
            loss_coef: Balance loss coefficient.

        Returns:
            output_list: List of tensors, each [seq_len, bs, embed_dim],
                         in the same order as x_list.
            total_loss: Scalar (balance loss + diversity loss).
        """
        assert len(x_list) == len(modality_labels), (
            f"Got {len(x_list)} inputs but {len(modality_labels)} labels"
        )

        output_list = []
        all_gates = []
        balance_loss = torch.tensor(0.0, device=x_list[0].device)

        for x, mod_label in zip(x_list, modality_labels):
            #Transpose to batch-first
            x_bf = x.transpose(0, 1)  # [bs, seq_len, D]

            if torch.isnan(x_bf).any():
                return None, None

            #Route through modality-specific router
            router = self._get_router(mod_label)
            gates, load = router(x_bf, train=train)
            all_gates.append(gates)

            #Balance loss for this router
            balance_loss = balance_loss + (
                self._cv_squared(gates.sum(0)) + self._cv_squared(load)
            )

            #Dispatch to shared experts
            dispatcher = SparseDispatcher3D(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x_bf)

            expert_outputs = [
                self.experts[i](expert_inputs[i])
                for i in range(self.num_experts)
            ]

            combined = dispatcher.combine(expert_outputs)
            # combined: [bs, seq_len, D]

            output_list.append(combined.transpose(0, 1))  # [seq_len, bs, D]

        balance_loss = balance_loss * loss_coef
        diversity_loss = self._diversity_loss(all_gates) * self.diversity_coef
        total_loss = balance_loss + diversity_loss

        return output_list, total_loss