"""
FlexiSeqMoE: Robustified SeqMoE for clinical multimodal learning.

Drop-in replacement for SeqMoE that remains performant when test-time modality
combinations differ from training (e.g. trained on {TS, Text, CXR} but tested
with only {TS}).

Core mechanisms:
  1. Modality dropout (training): Randomly selects a subset of provided
     modalities as "reconstruction targets," computing a consistency loss
     between their learned proxy representation and their real pooled
     representation. This forces the expert pool to be individually useful
     for each modality, not just for the full combination.

  2. Learned proxy tokens: One ModalityProxy (a learned [proxy_seq_len, D]
     sequence) per modality type. At test time, absent modalities are replaced
     with their proxy, routed through the shared expert pool, and their balance
     contribution is counted — but proxy outputs are discarded from output_list.

  3. Cross-modal imputation (optional): Proxy tokens attend via cross-attention
     to all available modality sequences before routing, giving context-dependent
     proxies that can use available signal to better approximate the missing one.

  4. Reconstruction consistency loss (optional): Cosine dissimilarity between
     the pooled proxy representation and the pooled real (target-detached)
     representation. Trained without an extra expert pass — uses temporal_pool only.

Backward compatibility:
  FlexiSeqMoE(FlexiMoEConfig(modality_dropout_rate=0.0, use_modality_proxies=False))
  is behaviourally identical to SeqMoE with the same base parameters.

Usage example:
    config = FlexiMoEConfig(
        num_experts=8,
        embed_dim=128,
        moe_hidden_size=256,
        modality_types=['ts', 'txt', 'cxr'],
        top_k=2,
        modality_dropout_rate=0.3,
        use_modality_proxies=True,
        proxy_seq_len=8,
        use_cross_modal_imputation=False,
        reconstruction_coef=0.01,
    )
    moe = FlexiSeqMoE(config)

    # Training — all modalities provided, dropout / recon loss applied internally
    out_list, loss = moe(
        x_list=[proj_x_ts, proj_x_txt, proj_x_cxr],
        modality_labels=['ts', 'txt', 'cxr'],
        train=True,
    )

    # Test with only TS — proxies generated for txt and cxr internally
    out_list, loss = moe(
        x_list=[proj_x_ts],
        modality_labels=['ts'],
        train=False,
    )

    # In TransformerCrossEncoderLayer (zero-change call site):
    moe_out, balance_loss = self.moe(x_list, modality_labels=[m.lower() for m in modality])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fusemoe_multitask.moe import (
    SeqMoE,
    MoEConfig,
    SparseDispatcher3D,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class FlexiMoEConfig(MoEConfig):
    """Extended MoEConfig adding modality-dropout and proxy-token parameters.

    All base MoEConfig arguments are accepted unchanged. New arguments control
    the robustness mechanisms added by FlexiSeqMoE.

    Args:
        modality_dropout_rate (float): During training, each provided modality
            is independently selected as a reconstruction target with this
            probability. Reconstruction loss is computed for selected targets.
            Set to 0.0 to disable dropout entirely. Default 0.3.
        min_modalities (int): Minimum number of modalities guaranteed NOT to be
            selected as reconstruction targets in any forward call. Must be >= 1
            to ensure at least one real modality is always routed cleanly.
            Default 1.
        use_modality_proxies (bool): If True, create a ModalityProxy per
            modality type and route them for absent modalities. Proxy outputs are
            discarded from output_list but contribute to balance loss. Default True.
        proxy_seq_len (int): Token-sequence length for each ModalityProxy. Does
            not need to match the real modality's seq_len — proxy outputs are
            always discarded. Default 8.
        use_cross_modal_imputation (bool): If True, proxy tokens are refined via
            cross-attention over all real (available) modality sequences before
            routing. Requires use_modality_proxies=True. Default False.
        imputation_heads (int): Number of attention heads in CrossModalImputer.
            Must divide embed_dim. Default 4.
        reconstruction_coef (float): Coefficient for the reconstruction
            consistency loss term. Set to 0.0 (default) to disable completely —
            no temporal_pool calls are made for reconstruction when disabled.
    """

    def __init__(
        self,
        # ---- Pass-through to MoEConfig ----
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
        # ---- FlexiSeqMoE-specific ----
        modality_dropout_rate: float = 0.3,
        min_modalities: int = 1,
        use_modality_proxies: bool = True,
        proxy_seq_len: int = 8,
        use_cross_modal_imputation: bool = False,
        imputation_heads: int = 4,
        reconstruction_coef: float = 0.0,
    ):
        super().__init__(
            num_experts=num_experts,
            embed_dim=embed_dim,
            moe_hidden_size=moe_hidden_size,
            modality_types=modality_types,
            top_k=top_k,
            temporal_kernel=temporal_kernel,
            noisy_gating=noisy_gating,
            gating=gating,
            dropout=dropout,
            hidden_act=hidden_act,
            use_default_router=use_default_router,
            diversity_coef=diversity_coef,
            diversity_mode=diversity_mode,
        )
        if not (0.0 <= modality_dropout_rate <= 1.0):
            raise ValueError(
                f"modality_dropout_rate must be in [0, 1], got {modality_dropout_rate}"
            )
        if min_modalities < 1:
            raise ValueError(
                f"min_modalities must be >= 1, got {min_modalities}"
            )
        if use_cross_modal_imputation and not use_modality_proxies:
            raise ValueError(
                "use_cross_modal_imputation=True requires use_modality_proxies=True"
            )

        self.modality_dropout_rate = modality_dropout_rate
        self.min_modalities = min_modalities
        self.use_modality_proxies = use_modality_proxies
        self.proxy_seq_len = proxy_seq_len
        self.use_cross_modal_imputation = use_cross_modal_imputation
        self.imputation_heads = imputation_heads
        self.reconstruction_coef = reconstruction_coef


# ---------------------------------------------------------------------------
# ModalityProxy
# ---------------------------------------------------------------------------

class ModalityProxy(nn.Module):
    """Learned proxy token sequence representing an absent modality.

    Stores a fixed-length sequence of learned embedding vectors that serve as
    a stand-in when a modality is missing. The same proxy tokens are broadcast
    identically across all batch elements, making the proxy a prior over what
    the absent modality would have contributed.

    Initialization: proxy_tokens uses nn.init.normal_(std=0.02), matching the
    positional embedding convention in module.py (lines 369, 372).

    Args:
        proxy_seq_len (int): Number of position tokens in the proxy sequence.
        embed_dim (int): Embedding dimension — must match MoE embed_dim.
    """

    def __init__(self, proxy_seq_len: int, embed_dim: int):
        super().__init__()
        self.proxy_tokens = nn.Parameter(torch.empty(proxy_seq_len, embed_dim))
        nn.init.normal_(self.proxy_tokens, std=0.02)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Broadcast proxy tokens across the batch dimension.

        Args:
            batch_size: Number of samples in the current batch.
            device: Target device for the output tensor.

        Returns:
            Tensor of shape [proxy_seq_len, batch_size, embed_dim].
            All batch elements receive identical proxy token values.
        """
        # [proxy_seq_len, embed_dim] → [proxy_seq_len, 1, embed_dim] → expand
        tokens = self.proxy_tokens.to(device)
        return tokens.unsqueeze(1).expand(-1, batch_size, -1)


# ---------------------------------------------------------------------------
# CrossModalImputer
# ---------------------------------------------------------------------------

class CrossModalImputer(nn.Module):
    """Refines proxy token sequences via cross-attention over available modalities.

    When a modality is absent, its proxy tokens may be improved by attending to
    sequences that ARE available. For example, if CXR is absent, the CXR proxy
    can attend to TS and Text sequences to better approximate what a CXR
    representation would look like for this patient.

    Architecture:
        Query  = proxy_seq  (tokens to be refined)
        Key/V  = concat of all available modality sequences
        Output = residual-connected, layer-normalized refined proxy

    Uses nn.MultiheadAttention(batch_first=False) to match the seq-first tensor
    convention throughout this codebase ([seq, batch, embed]).

    Args:
        embed_dim (int): Embedding dimension.
        imputation_heads (int): Number of attention heads. Must divide embed_dim.
        dropout (float): Attention dropout rate.
    """

    def __init__(self, embed_dim: int, imputation_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=imputation_heads,
            dropout=dropout,
            batch_first=False,   # seq-first: [seq, batch, embed]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        proxy_seq: torch.Tensor,
        available_seqs: list,
    ) -> torch.Tensor:
        """Refine a proxy sequence by attending to all available modality sequences.

        Args:
            proxy_seq: [proxy_seq_len, batch_size, embed_dim] — initial proxy
                token sequence (seq-first).
            available_seqs: List of tensors each [seq_len_i, batch_size, embed_dim].
                These are the real modality sequences currently available.
                They become the keys and values for cross-attention.

        Returns:
            Refined proxy of shape [proxy_seq_len, batch_size, embed_dim].
            Returns proxy_seq unchanged if available_seqs is empty.
        """
        if not available_seqs:
            return proxy_seq

        # Concatenate all available modality sequences as context
        # context: [sum(seq_len_i), batch_size, embed_dim]
        context = torch.cat(available_seqs, dim=0)

        # Cross-attention: proxy attends to all available modalities
        # attn_output: [proxy_seq_len, batch_size, embed_dim]
        attn_output, _ = self.cross_attn(
            query=proxy_seq,
            key=context,
            value=context,
        )

        # Post-norm residual (consistent with encoder layers in module.py)
        return self.norm(proxy_seq + attn_output)


# ---------------------------------------------------------------------------
# FlexiSeqMoE
# ---------------------------------------------------------------------------

class FlexiSeqMoE(SeqMoE):
    """Robustified SeqMoE with modality dropout and learned proxy tokens.

    Trains to handle variable modality subsets at test time by:
      1. Randomly selecting a subset of provided modalities as "reconstruction
         targets" during training (modality dropout), and penalizing divergence
         between their proxy representation and their real representation.
      2. Using learned proxy token sequences to fill in for truly absent
         modalities, routing them through the shared expert pool so the balance
         loss and expert specialization still see the full modality set.
      3. Optionally refining proxies via cross-attention over available modalities
         (CrossModalImputer).
      4. Optionally applying a reconstruction consistency loss that aligns the
         pooled proxy representation with the pooled real representation.

    Inheritance from SeqMoE:
        - self.routers (nn.ModuleDict): per-modality ModalityRouter instances
        - self.experts (nn.ModuleList): shared TemporalExpertMLP pool
        - self.default_router (ModalityRouter or None): fallback router
        - self._get_router(label): router lookup with fallback
        - self._cv_squared(x): coefficient of variation squared
        - self._diversity_loss(gates_list): pairwise routing overlap loss

    New components:
        - self.modality_proxies (nn.ModuleDict): one ModalityProxy per modality
        - self.default_proxy (ModalityProxy | None): fallback proxy
        - self.imputer (CrossModalImputer | None): cross-attention imputer

    Key invariants maintained by forward():
        - len(output_list) == len(x_list) always
        - output_list[i].shape == x_list[i].shape always (seq_len preserved)
        - Proxy outputs are NEVER in output_list (only outputs for provided modalities)
        - Proxy balance contributions ARE counted in balance_loss
        - Diversity loss is computed from real modality gates ONLY
    """

    def __init__(self, config: FlexiMoEConfig):
        # Initializes self.routers, self.experts, self.default_router
        super().__init__(config)
        self.config = config  # overwrite with FlexiMoEConfig (has extra fields)

        # --- Proxy tokens: one per known modality ---
        if config.use_modality_proxies:
            self.modality_proxies = nn.ModuleDict({
                mod: ModalityProxy(config.proxy_seq_len, config.embed_dim)
                for mod in config.modality_types
            })
            # Fallback proxy parallels default_router
            self.default_proxy = (
                ModalityProxy(config.proxy_seq_len, config.embed_dim)
                if config.use_default_router else None
            )
        else:
            self.modality_proxies = nn.ModuleDict()
            self.default_proxy = None

        # --- Cross-modal imputer (optional) ---
        if config.use_cross_modal_imputation:
            # Validation in FlexiMoEConfig already ensures use_modality_proxies=True
            self.imputer = CrossModalImputer(
                embed_dim=config.embed_dim,
                imputation_heads=config.imputation_heads,
                dropout=config.dropout,
            )
        else:
            self.imputer = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_proxy(self, mod_label: str) -> ModalityProxy:
        """Look up the proxy for a modality type, with fallback.

        Parallel structure to SeqMoE._get_router().

        Args:
            mod_label: Lowercase modality label.

        Returns:
            ModalityProxy for this modality.

        Raises:
            ValueError: If label is unknown and no default_proxy configured.
        """
        if mod_label in self.modality_proxies:
            return self.modality_proxies[mod_label]
        elif self.default_proxy is not None:
            return self.default_proxy
        else:
            raise ValueError(
                f"Unknown modality '{mod_label}' and no default proxy configured. "
                f"Known types: {list(self.modality_proxies.keys())}"
            )

    def _select_reconstruction_targets(
        self,
        x_list: list,
        modality_labels: list,
    ):
        """Randomly select a subset of provided modalities as reconstruction targets.

        Each modality is selected independently with probability
        config.modality_dropout_rate. At least (n - (n - min_modalities))
        modalities remain non-targets, i.e. at most (n - min_modalities)
        targets are selected. If too many are selected, a random subset is
        de-selected until the constraint is satisfied.

        Args:
            x_list: List of [seq_len, bs, D] tensors (seq-first).
            modality_labels: Lowercase labels parallel to x_list.

        Returns:
            target_x (list): Subset of x_list chosen as targets.
            target_labels (list): Corresponding labels for target_x.
            non_target_x (list): Remaining (non-target) modality tensors.
        """
        n = len(x_list)
        max_targets = max(0, n - self.config.min_modalities)

        if max_targets == 0 or self.config.modality_dropout_rate == 0.0:
            return [], [], list(x_list)

        # Per-modality Bernoulli draw using PyTorch RNG (reproducible with seed)
        draws = torch.rand(n)
        is_target = [draws[i].item() < self.config.modality_dropout_rate
                     for i in range(n)]

        # Clamp number of targets to max_targets
        target_indices = [i for i, t in enumerate(is_target) if t]
        if len(target_indices) > max_targets:
            # Randomly de-select extras
            perm = torch.randperm(len(target_indices)).tolist()
            keep_as_target = set(target_indices[perm[i]] for i in range(max_targets))
            is_target = [i in keep_as_target for i in range(n)]

        target_x = [x_list[i] for i, t in enumerate(is_target) if t]
        target_labels = [modality_labels[i] for i, t in enumerate(is_target) if t]
        non_target_x = [x_list[i] for i, t in enumerate(is_target) if not t]

        return target_x, target_labels, non_target_x

    def _build_proxy(
        self,
        mod_label: str,
        batch_size: int,
        device: torch.device,
        available_x_list: list,
    ) -> torch.Tensor:
        """Construct a proxy sequence for an absent modality.

        Step 1: Retrieve the learned proxy tokens for this modality and broadcast
                to the current batch size → [proxy_seq_len, bs, D] (seq-first).
        Step 2 (optional): If use_cross_modal_imputation=True and available_x_list
                is non-empty, refine the proxy via cross-attention over available
                real modality sequences.

        Args:
            mod_label: Lowercase label for the absent modality.
            batch_size: Number of samples in the current batch.
            device: Device for tensor placement.
            available_x_list: List of REAL modality tensors currently available
                (seq-first [seq_len_i, bs, D]). Used as cross-attention context.

        Returns:
            Proxy tensor of shape [proxy_seq_len, batch_size, embed_dim] (seq-first).
        """
        proxy_module = self._get_proxy(mod_label)
        # [proxy_seq_len, batch_size, embed_dim], seq-first
        proxy_seq = proxy_module(batch_size, device)

        if self.imputer is not None and available_x_list:
            proxy_seq = self.imputer(proxy_seq, available_x_list)

        return proxy_seq

    def _route_single(
        self,
        x_bf: torch.Tensor,
        mod_label: str,
        train: bool,
    ):
        """Route a single modality (batch-first) through the shared expert pool.

        Extracted from SeqMoE.forward's inner loop so both real and proxy
        sequences can be routed with identical code paths.

        Args:
            x_bf: [batch_size, seq_len, embed_dim] — batch-first input tensor.
            mod_label: Lowercase modality label used to select the router.
            train: Whether to apply noisy gating.

        Returns:
            output_bf: [batch_size, seq_len, embed_dim] — routed output, batch-first.
            gates: [batch_size, num_experts] — sparse gate weights.
            balance_contribution: Scalar tensor — CV²(importance) + CV²(load).
        """
        router = self._get_router(mod_label)
        gates, load = router(x_bf, train=train)

        balance = self._cv_squared(gates.sum(0)) + self._cv_squared(load)

        dispatcher = SparseDispatcher3D(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_bf)
        expert_outputs = [
            self.experts[i](expert_inputs[i])
            for i in range(self.num_experts)
        ]
        output_bf = dispatcher.combine(expert_outputs)

        return output_bf, gates, balance

    def _compute_reconstruction_loss(
        self,
        target_x: list,
        target_labels: list,
        batch_size: int,
        device: torch.device,
        non_target_x: list,
    ) -> torch.Tensor:
        """Cosine dissimilarity between pooled real and proxy representations.

        For each reconstruction target modality:
            1. Pool the real (target) sequence using the modality router's
               temporal_pool: real_pooled = router.temporal_pool(real_bf)  [bs, D]
            2. Build a proxy for this modality (using non-target modalities as
               cross-attention context if imputation is enabled).
            3. Pool the proxy: proxy_pooled = router.temporal_pool(proxy_bf)  [bs, D]
            4. loss += 1 - cosine_similarity(real_pooled.detach(), proxy_pooled).mean()

        The real side is detached so gradients only flow into proxy parameters
        (proxy tokens and imputer weights), not into the encoder feature path.
        No extra expert routing pass is needed — only temporal_pool is called.

        Args:
            target_x: List of [seq_len, bs, D] tensors (seq-first) for targets.
            target_labels: Labels for target_x, in matching order.
            batch_size: Batch size (for proxy construction).
            device: Device (for proxy construction and zero-tensor init).
            non_target_x: List of [seq_len, bs, D] tensors for non-target
                modalities — used as cross-attention context in _build_proxy.

        Returns:
            Scalar loss averaged over reconstruction targets. Returns tensor(0.0)
            on the given device if target_labels is empty.
        """
        if not target_labels:
            return torch.tensor(0.0, device=device)

        total_loss = torch.tensor(0.0, device=device)

        for real_seq, lbl in zip(target_x, target_labels):
            # Build proxy for this target modality
            # Use non-target modalities as cross-attention context (if imputer enabled)
            proxy_seq = self._build_proxy(lbl, batch_size, device, non_target_x)

            # Transpose to batch-first for temporal_pool
            real_bf = real_seq.transpose(0, 1)     # [bs, seq_len, D]
            proxy_bf = proxy_seq.transpose(0, 1)   # [bs, proxy_seq_len, D]

            router = self._get_router(lbl)

            # Pool to [bs, D]; real side detached (teacher)
            real_pooled = router.temporal_pool(real_bf).detach()
            proxy_pooled = router.temporal_pool(proxy_bf)

            # Cosine dissimilarity: 1 - cos_sim, averaged over batch
            cos_sim = F.cosine_similarity(real_pooled, proxy_pooled, dim=-1)
            total_loss = total_loss + (1.0 - cos_sim).mean()

        return total_loss / len(target_labels)

    # ------------------------------------------------------------------
    # Public forward (overrides SeqMoE.forward)
    # ------------------------------------------------------------------

    def forward(
        self,
        x_list: list,
        modality_labels: list,
        train: bool = True,
        loss_coef: float = 1e-2,
    ):
        """Full FlexiSeqMoE forward pass.

        Pipeline:
            Step 0: Validate inputs; guard empty x_list.
            Step 1: NaN check — return (None, None) on corrupt inputs (matches SeqMoE).
            Step 2: [Train only] Select reconstruction targets via modality dropout.
            Step 3: Route ALL provided modalities through shared experts.
            Step 4: [If use_modality_proxies] Build + route proxies for absent
                    modalities; proxy outputs discarded, their balance counted.
            Step 5: Compute losses: balance + diversity (real only) + reconstruction.
            Step 6: Assemble output_list in original modality order.
            Step 7: Return (output_list, total_loss).

        Args:
            x_list: List of [seq_len, batch_size, embed_dim] tensors (seq-first),
                    one per modality currently available. Different modalities may
                    have different seq_len values.
            modality_labels: Lowercase modality labels parallel to x_list,
                    e.g. ['ts', 'txt'] or ['ts', 'cxr', 'txt'].
            train: If True, applies modality dropout and noisy gating. If False,
                   no reconstruction targets selected; proxies still built for
                   absent modalities (if use_modality_proxies=True).
            loss_coef: Scaling coefficient for the balance loss term.

        Returns:
            output_list: List of [seq_len, bs, embed_dim] in the same order as
                         the original x_list. len(output_list) == len(x_list).
                         None if NaN detected in inputs.
            total_loss: Scalar combining balance, diversity, and reconstruction
                        losses. None if NaN detected in inputs.
        """
        # ------------------------------------------------------------------ #
        # Step 0: Validate inputs
        # ------------------------------------------------------------------ #
        assert len(x_list) == len(modality_labels), (
            f"Got {len(x_list)} inputs but {len(modality_labels)} labels"
        )
        if len(x_list) == 0:
            return [], torch.tensor(0.0)

        original_labels = list(modality_labels)
        batch_size = x_list[0].shape[1]   # x[0]: [seq_len, bs, D]
        device = x_list[0].device

        # ------------------------------------------------------------------ #
        # Step 1: NaN guard
        # ------------------------------------------------------------------ #
        for x in x_list:
            if torch.isnan(x).any():
                return None, None

        # ------------------------------------------------------------------ #
        # Step 2: Select reconstruction targets (training only)
        # ------------------------------------------------------------------ #
        target_x, target_labels, non_target_x = [], [], list(x_list)
        if train and self.config.modality_dropout_rate > 0.0:
            target_x, target_labels, non_target_x = (
                self._select_reconstruction_targets(x_list, list(modality_labels))
            )

        # ------------------------------------------------------------------ #
        # Step 3: Route ALL provided modalities (real inputs, no proxy here)
        # ------------------------------------------------------------------ #
        real_outputs_dict = {}   # label → seq-first output [seq_len, bs, D]
        real_gates_list = []     # for diversity loss (real modalities only)
        balance_loss = torch.tensor(0.0, device=device)

        for x_seq, lbl in zip(x_list, modality_labels):
            x_bf = x_seq.transpose(0, 1)    # [bs, seq_len, D]
            output_bf, gates, balance_contrib = self._route_single(
                x_bf, lbl, train
            )
            real_outputs_dict[lbl] = output_bf.transpose(0, 1)  # back to seq-first
            real_gates_list.append(gates)
            balance_loss = balance_loss + balance_contrib

        # ------------------------------------------------------------------ #
        # Step 4: Build + route proxies for absent modalities
        # ------------------------------------------------------------------ #
        # "Absent" = in config.modality_types but NOT provided in x_list this call
        if self.config.use_modality_proxies:
            provided_set = set(modality_labels)
            absent_modalities = [
                m for m in self.config.modality_types if m not in provided_set
            ]
            for mod_label in absent_modalities:
                proxy_seq = self._build_proxy(
                    mod_label=mod_label,
                    batch_size=batch_size,
                    device=device,
                    available_x_list=list(x_list),   # all real seqs as context
                )
                proxy_bf = proxy_seq.transpose(0, 1)   # [bs, proxy_seq_len, D]
                _proxy_out_bf, _proxy_gates, proxy_balance = self._route_single(
                    proxy_bf, mod_label, train
                )
                # Proxy outputs discarded; only balance contribution is kept
                balance_loss = balance_loss + proxy_balance

        # ------------------------------------------------------------------ #
        # Step 5: Compute losses
        # ------------------------------------------------------------------ #
        # 5a. Balance loss (covers real + proxy routing)
        scaled_balance = balance_loss * loss_coef

        # 5b. Diversity loss — real modality gates only
        if real_gates_list:
            diversity_loss = (
                self._diversity_loss(real_gates_list) * self.config.diversity_coef
            )
        else:
            diversity_loss = torch.tensor(0.0, device=device)

        # 5c. Reconstruction consistency loss (optional)
        if (train
                and self.config.reconstruction_coef > 0.0
                and target_labels
                and self.config.use_modality_proxies):
            recon_loss = self._compute_reconstruction_loss(
                target_x=target_x,
                target_labels=target_labels,
                batch_size=batch_size,
                device=device,
                non_target_x=non_target_x,
            ) * self.config.reconstruction_coef
        else:
            recon_loss = torch.tensor(0.0, device=device)

        total_loss = scaled_balance + diversity_loss + recon_loss

        # ------------------------------------------------------------------ #
        # Step 6: Assemble output_list in original label order
        # ------------------------------------------------------------------ #
        # All originally-provided modalities were routed in Step 3 and are in
        # real_outputs_dict. Output shape is guaranteed to match input shape.
        output_list = [real_outputs_dict[lbl] for lbl in original_labels]

        # ------------------------------------------------------------------ #
        # Step 7: Return
        # ------------------------------------------------------------------ #
        return output_list, total_loss
