"""Elastic Weight Consolidation (EWC) state for the continual-learning
pipeline. Mirrors the reference implementation at
``/cis/home/schaud35/SEED/src/approach/ewc.py`` (Kirkpatrick et al.,
"Overcoming catastrophic forgetting in neural networks", PNAS 2017,
arXiv:1612.00796) but scoped to the **expert weight matrices** of every
``SeqMoE`` block: ``temporal_conv.weight``, ``fc1.weight``, ``fc2.weight``.

This is the same set of weights that ``--cl_method ours`` SVD-truncates
between stages and that ``--cl_method lora`` adds LoRA adapters to. In
``--cl_method ewc`` mode those weights are *continually trainable* across
stages; the EWC quadratic penalty regularizes them toward their post-task
snapshot, weighted by the diagonal Fisher information matrix.

Lifecycle:

  * Construct ``EWCState(model)`` once at the start of training. Initial
    ``older_params`` is the random-init expert weights and ``fisher`` is
    zero, so the regularizer is zero on stage 0 (no anchor yet).

  * After stage ``s`` finishes (and best-on-val is restored):
      1. ``snapshot_older_params(model)`` -- copy the trained weights as
         the new anchor for stage ``s+1``.
      2. Compute Fisher on stage's training data via
         :py:func:`src.continual.cl_train._compute_fisher_for_stage`.
      3. ``merge_fisher(curr_fisher, alpha)`` -- fuse with running Fisher.

  * During stage ``s+1`` training, add ``λ · regularizer(model)`` to each
    batch's loss before backward.
"""

from typing import Callable, Dict, Iterable, Tuple

import torch
from torch import nn

from src.continual.cl_moe import find_seq_moes


# Suffixes of named_parameters that mark "expert weight matrices" -- the
# subset EWC regularizes. Matches the set we SVD-truncate in the
# `cl_method=ours` pipeline and that LoRAAdapter targets in `cl_method=lora`.
_EWC_TARGET_SUFFIXES = (
    '.temporal_conv.weight',
    '.fc1.weight',
    '.fc2.weight',
)

# Suffixes for the router-weight matrices of a single ``ModalityRouter`` --
# its ``w_gate`` and ``w_noise`` Parameters. Used by ``iter_router_weights``
# when ``--router_expansion`` is disabled in EWC mode (one shared router
# across stages, EWC-regularized like the experts).
_EWC_ROUTER_SUFFIXES = (
    '.w_gate',
    '.w_noise',
)


def iter_expert_weights(model):
    """Yield ``(name, parameter)`` pairs for every expert weight matrix
    that EWC regularizes. The selection rule is: parameter name contains
    ``.moe.experts.`` (so it lives inside a SeqMoE expert) **and** ends
    with one of ``temporal_conv.weight``, ``fc1.weight``, ``fc2.weight``.
    Biases and LayerNorm parameters are excluded by design.
    """
    for name, param in model.named_parameters():
        if '.moe.experts.' not in name:
            continue
        if not any(name.endswith(s) for s in _EWC_TARGET_SUFFIXES):
            continue
        yield name, param


def iter_router_weights(model):
    """Yield ``(name, parameter)`` pairs for the ``w_gate`` and ``w_noise``
    matrices of the single shared task-0 router on every
    ``PerTaskModalityRouter`` wrapper. Used by EWC when
    ``--router_expansion`` is disabled: only ``task_routers[0]`` exists
    (no per-stage expansion), and EWC regularizes its weights alongside
    the expert weights.

    Selection rule: parameter name contains ``.task_routers.0.`` AND ends
    with ``.w_gate`` or ``.w_noise``. ``task_routers.<i>.`` for ``i > 0``
    is *excluded* -- if those exist (router_expansion=True), they're
    handled by the per-stage freeze/unfreeze logic, not EWC.
    """
    for name, param in model.named_parameters():
        if '.task_routers.0.' not in name:
            continue
        if not any(name.endswith(s) for s in _EWC_ROUTER_SUFFIXES):
            continue
        yield name, param


def iter_expert_and_router_weights(model):
    """Yield expert + (single-shared-)router weights for the no-expansion
    EWC mode. EWCState constructed with this iterator regularizes both
    sets of params jointly with one shared λ.
    """
    yield from iter_expert_weights(model)
    yield from iter_router_weights(model)


def iter_encoder_target_weights(encoders):
    """Yield ``(name, param)`` pairs for encoder ``nn.Linear`` and
    ``nn.Conv1d(kernel_size=1)`` weight matrices that EWC should regularize
    when ``--cl_target=moe_and_encoder``. Mirrors the layer scope used by
    ``apply_lowrank_to_encoders`` in ``src/analysis/eval_lowrank_experts.py``
    and by the ``ours``/``lora`` encoder paths.

    Encoders shared across task keys (under ``--shared_modality_encoders``)
    are deduplicated by ``id(encoder)`` so each underlying weight is yielded
    once. Yielded names are prefixed with ``__encoder__.<id_hex>.`` so they
    cannot collide with regular model ``named_parameters`` paths in
    ``EWCState``'s ``older_params`` / ``fisher`` dicts.

    Bias tensors and LayerNorm weights are excluded by design (matches the
    EWC scope on the MoE side: only weight matrices, not biases / norms).
    """
    seen_enc = set()
    seen_param = set()
    for enc in encoders.values():
        if id(enc) in seen_enc:
            continue
        seen_enc.add(id(enc))
        prefix = f'__encoder__.{id(enc):x}.'
        for name, m in enc.named_modules():
            is_linear = isinstance(m, nn.Linear)
            is_conv1d_k1 = (isinstance(m, nn.Conv1d)
                            and tuple(m.kernel_size) == (1,))
            if not (is_linear or is_conv1d_k1):
                continue
            if not hasattr(m, 'weight') or m.weight is None:
                continue
            if id(m.weight) in seen_param:
                continue
            seen_param.add(id(m.weight))
            yield prefix + name + '.weight', m.weight


def make_combined_target_iterator(base_iterator: Callable, encoders):
    """Build a callable ``f(model) -> iter[(name, param)]`` that yields
    from ``base_iterator(model)`` first, then from
    ``iter_encoder_target_weights(encoders)``.

    The encoders dict is captured by closure so the returned iterator has
    the same ``(model,)`` call signature as the standalone iterators above
    and can be plugged into ``EWCState(model, target_iterator=...)`` /
    ``_compute_fisher_for_stage(... target_iterator=...)`` without any
    further plumbing.
    """
    def _it(model):
        yield from base_iterator(model)
        yield from iter_encoder_target_weights(encoders)
    _it.__name__ = f'combined_{base_iterator.__name__}_with_encoders'
    return _it


class EWCState:
    """Holds the running EWC state: per-target ``fisher`` (diagonal Fisher
    matrix) and ``older_params`` (the snapshot the regularizer pulls toward).
    Both indexed by full ``named_parameters`` path.

    The set of "target" parameters EWC regularizes is configurable via
    ``target_iterator``: a callable ``f(model) -> iter[(name, param)]``.
    Defaults to :py:func:`iter_expert_weights` (the standard expert-only
    EWC). For ``--router_expansion`` disabled, pass
    :py:func:`iter_expert_and_router_weights` so the regularizer also
    constrains the shared router's ``w_gate``/``w_noise``.
    """

    def __init__(self, model, target_iterator: Callable = iter_expert_weights):
        self._iter = target_iterator
        self.older_params: Dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in self._iter(model)
        }
        self.fisher: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p) for n, p in self._iter(model)
        }

    @property
    def target_iterator(self) -> Callable:
        return self._iter

    def snapshot_older_params(self, model):
        """Replace ``older_params`` with the model's current target weights.
        Called at the end of each stage (after best-on-val is restored)
        so the next stage's regularizer pulls toward the just-trained
        state. Detaches and clones so the snapshot is independent of
        future updates.
        """
        new_snap = {n: p.detach().clone() for n, p in self._iter(model)}
        # Zero-init Fisher entries for any new keys (defensive, in case
        # the architecture grows -- shouldn't happen for ewc but harmless).
        for n, p in new_snap.items():
            if n not in self.fisher:
                self.fisher[n] = torch.zeros_like(p)
        self.older_params = new_snap

    def merge_fisher(self, curr_fisher: Dict[str, torch.Tensor], alpha: float):
        """Merge ``curr_fisher`` into ``self.fisher`` with weight ``alpha``:

        ``self.fisher[n] = α · self.fisher[n] + (1 − α) · curr_fisher[n]``

        ``alpha = 0`` keeps only the current task's Fisher (forgets prior).
        ``alpha = 1`` keeps the prior Fisher unchanged (ignores current).
        ``alpha = 0.5`` (the reference default) is a 50/50 fusion.
        """
        for n, curr in curr_fisher.items():
            if n in self.fisher:
                old = self.fisher[n].to(curr.device)
                self.fisher[n] = alpha * old + (1.0 - alpha) * curr
            else:
                self.fisher[n] = curr.detach().clone()

    def regularizer(self, model):
        """Compute the scalar EWC penalty
        ``Σ_n  fisher[n] · (θ_n − θ_n^old)² / 2``
        summed over the configured target parameters. Gradient flows back
        to the current parameters ``θ_n`` (Fisher and ``older_params``
        are detached constants).
        """
        loss = None
        any_param = None
        for n, p in self._iter(model):
            any_param = p
            if n not in self.fisher or n not in self.older_params:
                continue
            f = self.fisher[n].to(p.device)
            old = self.older_params[n].to(p.device)
            term = 0.5 * (f * (p - old).pow(2)).sum()
            loss = term if loss is None else (loss + term)
        if loss is None:
            device = (any_param.device if any_param is not None
                      else next(model.parameters()).device)
            return torch.zeros((), device=device)
        return loss

    def num_params_tracked(self):
        """Total number of scalar entries in ``older_params`` (= total
        number of expert weight values being regularized). Useful for
        logging and sanity checks."""
        return sum(p.numel() for p in self.older_params.values())

    def device_to(self, device):
        """Move ``older_params`` and ``fisher`` tensors to ``device``."""
        self.older_params = {n: p.to(device) for n, p in self.older_params.items()}
        self.fisher = {n: f.to(device) for n, f in self.fisher.items()}
