"""FlexMoE drop-in replacement for MULTCrossModel.

Wraps the original Flex-MoE backbone (src/flexmoe/models.py, ported verbatim
from https://github.com/UNITES-Lab/flex-moe) so it can be plugged into the
clinical-highmmt pipeline by flipping `args.fusion_model='flexmoe'`.

Input/output contract mirrors `src.fusemoe.MULTCrossModel`:
    forward(multi_modality_data: Dict[str, Tensor], task=..., use_recon=..., ...)
        -> (logits, gate_loss)              when use_recon=False
        -> (logits, None, gate_loss)        when use_recon=True
Each modality tensor has shape (B, seq_len, embed_dim). `gate_loss` is the
Flex-MoE router regularizer (sum of per-gate losses); the outer training loop
scales it by `args.balance_loss_coef` — same coefficient as MULTCrossModel's
balance_loss.
"""
from itertools import combinations
import torch
import torch.nn as nn

from .models import FlexMoE as _FlexMoECore


def _build_combination_to_index(num_modalities):
    """Flex-MoE data-pipeline convention: full combination -> index 0.
    Iterates combinations from size n down to 1 (matches Flex-MoE/data.py:
    get_modality_combinations)."""
    all_combs = []
    for r in range(num_modalities, 0, -1):
        all_combs.extend(combinations(range(num_modalities), r))
    return {tuple(sorted(c)): idx for idx, c in enumerate(all_combs)}


class FlexMoE(nn.Module):
    def __init__(self, args, device, modalities, modalities_per_task,
                 num_classes=1, latent_dim=512, modeltype=None, **kwargs):
        super().__init__()
        self.args = args
        self.device = device
        self.modalities = modalities
        self.modalities_per_task = modalities_per_task
        self.num_modalities = len(modalities)
        self.embed_dim = args.embed_dim
        self.task = args.task
        self.modeltype = modeltype if modeltype is not None else getattr(args, 'modeltype', None)

        # Sorted modality name list gives stable index assignment.
        self.modality_names = sorted([m.name for m in modalities])
        self.name_to_idx = {n: i for i, n in enumerate(self.modality_names)}

        # Full combination -> index 0 (matches original Flex-MoE data pipeline).
        self.combination_to_index = _build_combination_to_index(self.num_modalities)
        self.full_modality_index = 0

        num_experts = args.num_of_experts[0] if isinstance(args.num_of_experts, (list, tuple)) else args.num_of_experts
        top_k       = args.top_k[0]          if isinstance(args.top_k, (list, tuple))          else args.top_k
        num_routers = getattr(args, 'num_routers', 1)
        num_layers      = getattr(args, 'layers', 2)
        num_layers_pred = getattr(args, 'num_layers_pred', 2)

        self.core = _FlexMoECore(
            num_modalities   = self.num_modalities,
            full_modality_index = self.full_modality_index,
            num_patches      = args.tt_max,
            hidden_dim       = args.embed_dim,
            output_dim       = args.embed_dim,   # unused (head replaced below)
            num_layers       = num_layers,
            num_layers_pred  = num_layers_pred,
            num_experts      = num_experts,
            num_routers      = num_routers,
            top_k            = top_k,
            num_heads        = args.num_heads,
            dropout          = args.dropout,
        )
        # Override core's combination index (it uses size 1->n; we need n->1).
        self.core.combination_to_index = self.combination_to_index

        # Inputs are already positionally encoded upstream — disable pos_embed.
        del self.core.pos_embed
        self.core.pos_embed = None

        # Replace MLP head with Identity; clinical-highmmt uses its own to_logits
        # which can be swapped externally per-task (model.to_logits = ...).
        self.core.network[-1] = nn.Identity()

        feature_dim = args.embed_dim * self.num_modalities
        self.to_logits = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes),
        )

    # --- expert / full-modality bookkeeping -----------------------------------

    def _map_key_to_modality_name(self, key):
        """Map a raw input key (e.g. 'TS_IHM', 'cc_IHM') to one of self.modality_names."""
        base = key.split('_')[0]
        for name in self.modality_names:
            if key == name or base == name.split('_')[0]:
                return name
        return None

    def _derive_routing(self, multi_modality_data, batch_size):
        present = set()
        for k in multi_modality_data.keys():
            name = self._map_key_to_modality_name(k)
            if name is not None:
                present.add(self.name_to_idx[name])
        combo = tuple(sorted(present))
        expert_idx = self.combination_to_index.get(combo, self.full_modality_index)
        is_full = (len(combo) == self.num_modalities)
        expert_indices = torch.full((batch_size,), expert_idx, dtype=torch.long, device=self.device)
        return expert_indices, is_full

    # --- forward --------------------------------------------------------------

    def forward(self, multi_modality_data, task=None, mask=None, use_recon=False,
                get_latent=False, get_pre_logits=False, latents=None, source_mode=None,
                get_catted=False, unimodal=False, null_pvi=False, labels=None):
        any_tensor = next(iter(multi_modality_data.values()))
        batch_size, seq_len, _ = any_tensor.shape

        # Reorder inputs to self.modality_names; fill absent modalities with zeros
        # so the concatenate/split pipeline in the core sees a consistent layout.
        key_by_name = {}
        for k in multi_modality_data:
            name = self._map_key_to_modality_name(k)
            if name is not None:
                key_by_name[name] = k

        ordered_inputs = []
        for name in self.modality_names:
            if name in key_by_name:
                ordered_inputs.append(multi_modality_data[key_by_name[name]])
            else:
                ordered_inputs.append(torch.zeros(
                    batch_size, seq_len, self.embed_dim,
                    device=self.device, dtype=any_tensor.dtype,
                ))

        expert_indices, is_full_modality = self._derive_routing(multi_modality_data, batch_size)
        self.core.set_full_modality(is_full_modality)

        feat = self.core(*ordered_inputs,
                         expert_indices=expert_indices,
                         is_full_modality=is_full_modality)
        # feat: (B, embed_dim * num_modalities) after mean-pool + concat + Identity.

        if get_pre_logits or get_latent or get_catted:
            return feat

        logits = self.to_logits(feat)
        gate_loss = self.core.gate_loss()

        if use_recon:
            return logits, None, gate_loss
        return logits, gate_loss

    # Expose Flex-MoE utility methods for external callers (symmetry with original).
    def gate_loss(self):
        return self.core.gate_loss()

    def set_full_modality(self, is_full_modality):
        self.core.set_full_modality(is_full_modality)

    def assign_expert(self, combination):
        return self.combination_to_index.get(tuple(sorted(combination)))
