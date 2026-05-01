"""MoE training-time diagnostics.

Logs five-mechanism fingerprints per epoch to detect which failure mode
is driving MoE underutilization:

  (1) Near-dead init: ||MoE|| / ||residual||
  (2) Self-trapping:  ||grad_expert|| / ||grad_encoder||
  (3) Gate symmetry:  routing entropy + pairwise expert cosine
  (4) Residual solves task: (ablation AUC handled externally)
  (5) Weight decay death: expert weight norms over epochs
  (6) Expert starvation: per-expert load fraction (min vs max)

Usage in training script:

    from src.analysis.moe_diagnostics import MoEDiagnosticsLogger

    diag = MoEDiagnosticsLogger(log_dir=savedir)   # creates log files
    diag.register_hooks(model)

    for epoch in range(n_epochs):
        for batch in loader:
            ...
            loss.backward()
            diag.log_grad_norms(model, epoch)      # ONE call per epoch is enough
            optim.step()
        diag.log_epoch(model, epoch)               # at end of each epoch

    diag.close()
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# Forward-order stages inside a TransformerCrossEncoderLayer. Ordered exactly
# as they are invoked in `TransformerCrossEncoderLayer.forward` (module.py).
# Each entry is a name-fragment to match against `<layer>.layers.{N}.<stage>...`
_LAYER_FORWARD_STAGES = (
    "pre_self_attn_layer_norm",
    "self_attns",
    "post_self_attn_layer_norm",
    "pre_encoder_attn_layer_norm",
    "moe.routers",
    "moe.experts",
    "moe.output_norm",
    "moe",                       # fallback for any other moe.* params
    "pre_combine_residual_norm",
    "residual_gate",
    "cross_attn_1",
    "cross_attn_2",
    "post_encoder_attn_layer_norm",
    "pre_ffn_layer_norm",
    "fc1",
    "fc2",
)


def _layer_group_for_name(name: str) -> str:
    """Coarse group label for a model parameter, used to colocate related
    params on a single wandb line_series plot.

    Strategy: keep everything structural (the stage), drop per-instance
    indices so e.g. all experts' `fc1.weight` land in one group.

      trans_self_cross_ts_txt.layers.0.moe.experts.3.fc1.weight
        -> "layers.moe.experts.fc1.weight"
      trans_self_cross_ts_txt.layers.1.pre_self_attn_layer_norm.0.weight
        -> "layers.pre_self_attn_layer_norm.weight"
      foo.bar.weight (no `layers.`)
        -> "other/foo.bar.weight"
    """
    parts = name.split(".")
    if "layers" not in parts:
        return f"other/{name}"
    li = parts.index("layers")
    tail = parts[li + 2:]  # skip "layers", "{N}"
    # drop purely numeric components (modality idx, expert idx, etc.)
    tail = [p for p in tail if not p.isdigit()]
    return "layers." + ".".join(tail) if tail else "layers"


def _layer_sort_key(name: str):
    """Return (layer_idx, stage_idx, name) for a cross-encoder param name.

    Params not matching a `.layers.{N}.<stage>.` pattern sort to the end.
    """
    parts = name.split(".")
    layer_idx = 10**6
    stage_idx = 10**6
    if "layers" in parts:
        li = parts.index("layers")
        if li + 1 < len(parts):
            try:
                layer_idx = int(parts[li + 1])
            except ValueError:
                pass
            tail = ".".join(parts[li + 2:])
            for i, stage in enumerate(_LAYER_FORWARD_STAGES):
                if tail.startswith(stage):
                    stage_idx = i
                    break
    return (layer_idx, stage_idx, name)


class LayerwiseGradLogger:
    """Logs per-parameter gradient L2 norms, separately for the model and the
    upstream encoders, in forward-pass chronology. Writes two jsonl files:

        <log_dir>/layerwise_grads_model.jsonl
        <log_dir>/layerwise_grads_encoder.jsonl

    Each line is one epoch's dump. Call `log(model, encoder_dict, epoch)` once
    per epoch AFTER `loss.backward()`, BEFORE `optim.step()`.
    """

    def __init__(self, log_dir: str,
                 model_jsonl: str = "layerwise_grads_model.jsonl",
                 encoder_jsonl: str = "layerwise_grads_encoder.jsonl",
                 wandb_run=None):
        os.makedirs(log_dir, exist_ok=True)
        self._model_fh = open(os.path.join(log_dir, model_jsonl), "a", buffering=1)
        self._encoder_fh = open(os.path.join(log_dir, encoder_jsonl), "a", buffering=1)
        self._epoch_logged = -1
        self._wandb = wandb_run
        self._model_history: dict = {}
        self._encoder_history: dict = {}

    @torch.no_grad()
    def log(self, model, encoders, epoch: int, force: bool = False):
        if not force and self._epoch_logged == epoch:
            return
        self._epoch_logged = epoch

        # --- Model side: order by (layer_idx, stage_idx, name) ---
        model_items = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            model_items.append((name, float(p.grad.detach().float().norm().item())))
        model_items.sort(key=lambda kv: _layer_sort_key(kv[0]))
        model_payload = {"epoch": int(epoch), "order": "forward",
                         "grads": [{"name": n, "grad_norm": g} for n, g in model_items]}
        self._model_fh.write(json.dumps(model_payload) + "\n")

        # --- Encoder side: one sub-dict per encoder, definition order ---
        encoder_payload = {"epoch": int(epoch), "order": "definition", "encoders": {}}
        # encoders may be a dict {task: module} or a single nn.Module
        if isinstance(encoders, dict):
            iterable = encoders.items()
        else:
            iterable = [("_single", encoders)]
        for task, enc in iterable:
            if enc is None:
                continue
            items = []
            for name, p in enc.named_parameters():
                if p.grad is None:
                    continue
                items.append({"name": name,
                              "grad_norm": float(p.grad.detach().float().norm().item())})
            encoder_payload["encoders"][task] = items
        self._encoder_fh.write(json.dumps(encoder_payload) + "\n")

        # --- Mirror to wandb: line_series per group; scalars ONLY for singletons ---
        if self._wandb is not None:
            flat = {"epoch": int(epoch)}

            # --- Model side ---
            model_groups = {}  # {group: {series_name: grad_norm}}
            for n, g in model_items:
                group_name = _layer_group_for_name(n)
                model_groups.setdefault(group_name, {})[n] = g
            # Update history
            for group_name, series_dict in model_groups.items():
                hist = self._model_history.setdefault(group_name, {})
                for sname, val in series_dict.items():
                    h = hist.setdefault(sname, {"xs": [], "ys": []})
                    h["xs"].append(int(epoch))
                    h["ys"].append(val)
            # Emit
            for group_name, hist in self._model_history.items():
                if group_name not in model_groups:
                    continue  # not updated this epoch
                if len(hist) >= 2:
                    keys = sorted(hist.keys())
                    xs = [hist[k]["xs"] for k in keys]
                    ys = [hist[k]["ys"] for k in keys]
                    title = f"grad norms: {group_name}"
                    fig = _colored_multiline_plot(xs, ys, keys, title=title, xname="epoch")
                    try:
                        if fig is not None:
                            flat[f"layerwise_grads/model/plots/{group_name}"] = fig
                        else:
                            flat[f"layerwise_grads/model/plots/{group_name}"] = \
                                self._wandb.plot.line_series(
                                    xs=xs, ys=ys, keys=keys,
                                    title=title, xname="epoch",
                                )
                    except Exception:
                        pass
                else:
                    # Singleton group → log raw scalar
                    (only_name,) = hist.keys()
                    flat[f"layerwise_grads/model/{only_name}"] = hist[only_name]["ys"][-1]

            # --- Encoder side: one group per task ---
            for task, items in encoder_payload["encoders"].items():
                hist = self._encoder_history.setdefault(task, {})
                for it in items:
                    h = hist.setdefault(it["name"], {"xs": [], "ys": []})
                    h["xs"].append(int(epoch))
                    h["ys"].append(it["grad_norm"])
                if len(hist) >= 2:
                    keys = sorted(hist.keys())
                    xs = [hist[k]["xs"] for k in keys]
                    ys = [hist[k]["ys"] for k in keys]
                    title = f"encoder grad norms: {task}"
                    fig = _colored_multiline_plot(xs, ys, keys, title=title, xname="epoch")
                    try:
                        if fig is not None:
                            flat[f"layerwise_grads/encoder/plots/{task}"] = fig
                        else:
                            flat[f"layerwise_grads/encoder/plots/{task}"] = \
                                self._wandb.plot.line_series(
                                    xs=xs, ys=ys, keys=keys,
                                    title=title, xname="epoch",
                                )
                    except Exception:
                        pass
                elif len(hist) == 1:
                    (only_name,) = hist.keys()
                    flat[f"layerwise_grads/encoder/{task}/{only_name}"] = hist[only_name]["ys"][-1]

            try:
                self._wandb.log(flat)
            except Exception:
                pass

    def close(self):
        for fh in (self._model_fh, self._encoder_fh):
            try:
                fh.close()
            except Exception:
                pass


_EXPERT_CLASS_NAMES = {"TemporalExpertMLP", "FlexExpertMLP"}
_ROUTER_CLASS_NAMES = {"ModalityRouter", "NoisyTopKGate"}
_MOE_CLASS_NAMES = {"SeqMoE", "FlexSeqMoE", "MoE", "HierarchicalMoE"}
_LAYER_CLASS_NAMES = {"TransformerCrossEncoderLayer"}


def _module_class(m):
    return type(m).__name__


def _find_modules(model, class_names):
    for name, mod in model.named_modules():
        if _module_class(mod) in class_names:
            yield name, mod


def _safe_norm(t):
    if t is None:
        return 0.0
    return float(t.detach().float().norm().item())


def _colored_multiline_plot(xs, ys, keys, title, xname="epoch", yname="value"):
    """Build a multi-line chart with a distinct color per series.

    Returns a Plotly Figure (which wandb renders natively with colored
    series in the Workspace). Falls back to `None` when plotly isn't
    installed so callers can revert to `wandb.plot.line_series`.
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        return None
    palette = (
        px.colors.qualitative.Dark24
        + px.colors.qualitative.Light24
        + px.colors.qualitative.Alphabet
    )
    fig = go.Figure()
    for i, (xs_s, ys_s, name) in enumerate(zip(xs, ys, keys)):
        fig.add_trace(go.Scatter(
            x=xs_s, y=ys_s, mode="lines", name=str(name),
            line=dict(color=palette[i % len(palette)], width=1.3),
        ))
    fig.update_layout(
        title=title,
        xaxis_title=xname, yaxis_title=yname,
        legend=dict(font=dict(size=9)),
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
    )
    return fig


def _effective_rank(mat):
    """exp(entropy of normalized singular values). 1.0 = rank-1."""
    if mat is None or mat.numel() == 0 or mat.shape[0] < 2:
        return 0.0
    try:
        s = torch.linalg.svdvals(mat.float())
    except Exception:
        return 0.0
    s = s[s > 1e-8]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * (p + 1e-20).log()).sum()).item())


class MoEDiagnosticsLogger:
    """Collects and writes MoE diagnostic metrics during training."""

    def __init__(self, log_dir: str, jsonl_name: str = "moe_diag.jsonl",
                 text_name: str = "moe_diag.txt", grad_log_every_n_steps: int = 0,
                 wandb_run=None):
        """
        Args:
            log_dir: directory to write logs into. Created if missing.
            grad_log_every_n_steps: 0 = log first batch of each epoch only;
                                    N>0 = log every N optimizer steps.
            wandb_run: optional. If provided (pass the `wandb` module or a
                specific run), every jsonl payload is also sent to wandb
                under the `moe_diag/<kind>/` prefix, with the epoch as
                an `epoch` field so you can plot vs. epoch in the UI.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(log_dir, jsonl_name)
        self.text_path = os.path.join(log_dir, text_name)
        self.grad_every = grad_log_every_n_steps
        self._wandb = wandb_run
        # {group_name: {series_name: {"xs": [...], "ys": [...]}}}
        self._series_history: dict = {}

        self._jsonl_fh = open(self.jsonl_path, "a", buffering=1)
        self._text_fh = open(self.text_path, "a", buffering=1)

        self._expert_outputs = {}     # layer_name -> {expert_idx -> tensor}
        self._router_outputs = {}     # router_name -> list of gate tensors per call
        self._hook_handles = []
        self._grad_logged_this_epoch = -1

    # ---------- hook registration ----------

    def register_hooks(self, model):
        """Attach forward hooks to experts and routers. Idempotent."""
        self.remove_hooks()

        # Experts
        for name, mod in _find_modules(model, _EXPERT_CLASS_NAMES):
            parts = name.split(".")
            try:
                layer_key = ".".join(parts[: parts.index("experts")])
                expert_idx = int(parts[parts.index("experts") + 1])
            except (ValueError, IndexError):
                layer_key = name
                expert_idx = 0
            self._expert_outputs.setdefault(layer_key, {})

            def _mk(lk, ei):
                def _hook(_m, _inp, out):
                    if isinstance(out, tuple):
                        out = out[0]
                    self._expert_outputs[lk][ei] = out.detach()
                return _hook
            self._hook_handles.append(mod.register_forward_hook(_mk(layer_key, expert_idx)))

        # Routers
        for name, mod in _find_modules(model, _ROUTER_CLASS_NAMES):
            self._router_outputs.setdefault(name, [])

            def _mk_r(nm):
                def _hook(_m, _inp, out):
                    if isinstance(out, tuple):
                        gates = out[0]
                    else:
                        gates = out
                    if torch.is_tensor(gates):
                        self._router_outputs[nm].append(gates.detach())
                return _hook
            self._hook_handles.append(mod.register_forward_hook(_mk_r(name)))

    def remove_hooks(self):
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []

    def _clear_caches(self):
        for layer_key in list(self._expert_outputs.keys()):
            self._expert_outputs[layer_key].clear()
        for nm in list(self._router_outputs.keys()):
            self._router_outputs[nm].clear()

    # ---------- metric collection ----------

    @torch.no_grad()
    def log_grad_norms(self, model, epoch, force=False):
        """Call once per epoch AFTER loss.backward(), BEFORE optim.step().

        Categorizes parameters by substring in the fully-qualified name:
          'experts' -> expert params
          'gate' or 'router' or 'w_gate' or 'w_noise' -> gate params
          'residual_gate' -> alpha gate
          everything else under the cross encoder -> cross-encoder params
          everything outside the cross encoder -> encoder params (upstream feature extractors)
        """
        if not force:
            if self.grad_every == 0 and self._grad_logged_this_epoch == epoch:
                return
            if self.grad_every > 0:
                # caller controls cadence
                pass
        self._grad_logged_this_epoch = epoch

        bucket = {"expert": [], "router_gate": [], "residual_gate": [],
                  "cross_encoder_other": [], "upstream_encoder": []}
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            n_lower = name.lower()
            gn = p.grad.detach().float().norm().item()
            if "residual_gate" in n_lower:
                bucket["residual_gate"].append(gn)
            elif "experts" in n_lower:
                bucket["expert"].append(gn)
            elif "w_gate" in n_lower or "w_noise" in n_lower or "gate" in n_lower or "router" in n_lower:
                bucket["router_gate"].append(gn)
            elif "trans_self_cross" in n_lower or "cross_attn" in n_lower:
                bucket["cross_encoder_other"].append(gn)
            else:
                bucket["upstream_encoder"].append(gn)

        payload = {"kind": "grad", "epoch": int(epoch)}
        for k, v in bucket.items():
            payload[f"grad_mean/{k}"] = float(np.mean(v)) if v else 0.0
            payload[f"grad_count/{k}"] = int(len(v))
        exp_m = payload["grad_mean/expert"]
        enc_m = payload["grad_mean/upstream_encoder"]
        payload["grad_ratio_expert_over_encoder"] = (
            exp_m / enc_m if enc_m > 1e-20 else float("inf") if exp_m > 0 else 0.0
        )
        self._write(payload)

    @torch.no_grad()
    def log_epoch(self, model, epoch):
        """Call at the end of each epoch — reads hook caches from the MOST RECENT
        forward pass (e.g. the last training batch or an eval pass)."""
        payload = {"kind": "epoch", "epoch": int(epoch)}

        # (5) Expert weight norms + (1 supporting) expert weight stats
        expert_wnorms = {}
        for name, p in model.named_parameters():
            if "experts" in name and p.ndim >= 1:
                expert_wnorms[name] = p.detach().float().norm().item()
        if expert_wnorms:
            payload["expert_wnorm/mean"] = float(np.mean(list(expert_wnorms.values())))
            payload["expert_wnorm/min"] = float(np.min(list(expert_wnorms.values())))
            payload["expert_wnorm/max"] = float(np.max(list(expert_wnorms.values())))

        # Effective rank of expert weight matrices (fc1.weight, fc2.weight,
        # temporal_conv.weight reshaped 2D). Key: layer/expert_idx/wname.
        for name, p in model.named_parameters():
            if "experts" not in name or not name.endswith(".weight"):
                continue
            W = p.detach().float()
            if W.dim() == 2:
                mat = W
            elif W.dim() == 3:
                mat = W.reshape(W.shape[0], -1)
            else:
                continue
            parts = name.split(".")
            try:
                i = parts.index("experts")
                layer_key = ".".join(parts[:i])
                expert_idx = parts[i + 1]
                wname = ".".join(parts[i + 2:]).replace(".weight", "")
            except (ValueError, IndexError):
                layer_key, expert_idx, wname = name, "?", "w"
            payload[f"{layer_key}/expert{expert_idx}/{wname}/weight_effective_rank"] = _effective_rank(mat)
            payload[f"{layer_key}/expert{expert_idx}/{wname}/weight_max_rank"] = int(min(mat.shape))

        # (3) Pairwise expert cosine per MoE layer  [from last forward]
        for layer_key, experts in sorted(self._expert_outputs.items()):
            if not experts:
                continue
            # Align to common token count
            idxs = sorted(experts.keys())
            tensors = []
            for k in idxs:
                t = experts[k]
                if isinstance(t, tuple):
                    t = t[0]
                tensors.append(t.reshape(-1, t.shape[-1]).float())
            min_tokens = min(t.shape[0] for t in tensors)
            if min_tokens < 2:
                continue
            tensors = [t[:min_tokens] for t in tensors]
            # average expert output across tokens: [K, D]
            means = torch.stack([t.mean(0) for t in tensors])
            means_n = F.normalize(means, dim=-1)
            sim = means_n @ means_n.t()
            K = len(idxs)
            if K > 1:
                off_diag = sim[~torch.eye(K, dtype=torch.bool, device=sim.device)]
                payload[f"{layer_key}/expert_cos_mean"] = float(off_diag.mean().item())
                payload[f"{layer_key}/expert_cos_max"] = float(off_diag.max().item())

            # Per-expert output norms (mean over tokens)
            per_norms = [float(t.norm(dim=-1).mean().item()) for t in tensors]
            payload[f"{layer_key}/expert_out_norm_mean"] = float(np.mean(per_norms))
            payload[f"{layer_key}/expert_out_norm_std"] = float(np.std(per_norms))

        # (3) + (6) Routing entropy and per-expert load from router hooks
        for router_name, gate_list in sorted(self._router_outputs.items()):
            if not gate_list:
                continue
            # Concatenate across calls in the captured forward
            gates = torch.cat([g.reshape(-1, g.shape[-1]) for g in gate_list], dim=0)
            if gates.numel() == 0:
                continue
            # entropy per-token under the gate distribution (clip to probs)
            p = gates.float()
            p_sum = p.sum(dim=-1, keepdim=True).clamp(min=1e-20)
            p = p / p_sum
            H = -(p * (p + 1e-20).log()).sum(dim=-1).mean().item()
            K = gates.shape[-1]
            payload[f"{router_name}/routing_entropy"] = float(H)
            payload[f"{router_name}/entropy_frac_of_uniform"] = float(H / math.log(K)) if K > 1 else 1.0
            load = (gates > 0).float().sum(dim=0)
            total = load.sum().clamp(min=1e-20)
            frac = (load / total).tolist()
            payload[f"{router_name}/load_min"] = float(min(frac))
            payload[f"{router_name}/load_max"] = float(max(frac))
            payload[f"{router_name}/load_fractions"] = [float(x) for x in frac]

        # (1) MoE output vs residual norms + effective rank (from layer's
        # _diag_residual / _diag_moe_output attributes, if the module exposed them)
        for name, mod in _find_modules(model, _LAYER_CLASS_NAMES):
            if not hasattr(mod, "_diag_residual") or not hasattr(mod, "_diag_moe_output"):
                continue
            res_list = getattr(mod, "_diag_residual")
            moe_list = getattr(mod, "_diag_moe_output")
            if res_list is None or moe_list is None:
                continue
            for i, (r, m) in enumerate(zip(res_list, moe_list)):
                r = r.float()
                m = m.float()
                r_flat = r.reshape(-1, r.shape[-1])
                m_flat = m.reshape(-1, m.shape[-1])
                payload[f"{name}/mod{i}/moe_norm_mean"] = float(m_flat.norm(dim=-1).mean().item())
                payload[f"{name}/mod{i}/residual_norm_mean"] = float(r_flat.norm(dim=-1).mean().item())
                denom = r_flat.norm(dim=-1).mean().clamp(min=1e-20)
                payload[f"{name}/mod{i}/moe_over_residual_ratio"] = float(
                    (m_flat.norm(dim=-1).mean() / denom).item()
                )
                payload[f"{name}/mod{i}/moe_effective_rank"] = _effective_rank(m_flat)
            alpha = getattr(mod, "_diag_alpha", None)
            if alpha is not None:
                payload[f"{name}/alpha"] = float(alpha)

        self._write(payload)
        self._clear_caches()

    # ---------- IO ----------

    def _write(self, payload: dict):
        self._jsonl_fh.write(json.dumps(payload) + "\n")
        # Human-readable summary
        kind = payload.get("kind", "?")
        ep = payload.get("epoch", "?")
        keys = [k for k in payload if k not in ("kind", "epoch")]
        # sort keys for stable text output
        keys.sort()
        line = f"[{kind} epoch={ep}] " + ", ".join(
            f"{k}={self._fmt(payload[k])}" for k in keys
        )
        self._text_fh.write(line + "\n")
        self._log_to_wandb(payload)

    # Suffixes that signal the last "/"-component is an aggregation label
    # (in which case the group is the prefix, not the suffix).
    _AGG_SUFFIXES = {"mean", "min", "max", "std", "count"}

    def _key_to_group_series(self, key: str):
        """Heuristic splitter: (group_name, series_label) for a flat key.

        Rules:
          - "a/b/mean"            -> ("a/b",   "mean")   (aggregation suffix)
          - "a/b/moe_norm_mean"   -> ("moe_norm_mean", "a/b")
          - "grad_mean/expert"    -> ("grad_mean", "expert")
          - "alpha"               -> ("alpha",  "_")
        """
        if "/" not in key:
            return key, "_"
        parts = key.split("/")
        if parts[-1] in self._AGG_SUFFIXES:
            return "/".join(parts[:-1]), parts[-1]
        return parts[-1], "/".join(parts[:-1])

    def _update_history(self, group: str, series: str, epoch: int, value: float):
        self._series_history.setdefault(group, {}) \
            .setdefault(series, {"xs": [], "ys": []})
        h = self._series_history[group][series]
        h["xs"].append(epoch)
        h["ys"].append(value)

    def _log_to_wandb(self, payload: dict):
        if self._wandb is None:
            return
        kind = payload.get("kind", "")
        prefix = f"moe_diag/{kind}/" if kind else "moe_diag/"
        ep = payload.get("epoch", None)
        if ep is None:
            return
        ep_int = int(ep)

        # Pass 1: determine (group, series) per key and update history.
        # Store (full_wandb_key, group, series, value) so we can decide later
        # which keys to log as scalars.
        entries = []  # list of (full_key, group, series, value)
        for k, v in payload.items():
            if k in ("kind", "epoch"):
                continue
            if isinstance(v, (int, float)):
                g, s = self._key_to_group_series(k)
                self._update_history(g, s, ep_int, float(v))
                entries.append((prefix + k, g, s, float(v)))
            elif isinstance(v, (list, tuple)):
                for i, vi in enumerate(v):
                    if isinstance(vi, (int, float)):
                        kparts = k.split("/")
                        g = kparts[-1]
                        s = ("/".join(kparts[:-1]) + f"/{i}").lstrip("/")
                        self._update_history(g, s, ep_int, float(vi))
                        entries.append((f"{prefix}{k}/{i}", g, s, float(vi)))

        # Pass 2: decide which groups are multi-series (plot) vs singletons (scalar).
        flat = {"epoch": ep_int}
        groups_seen = {g for _, g, _, _ in entries}
        multi_series_groups = {
            g for g in groups_seen if len(self._series_history.get(g, {})) >= 2
        }

        # Scalars only for singleton groups (no plot would be made for them).
        for full_key, g, _s, val in entries:
            if g not in multi_series_groups:
                flat[full_key] = val

        # Colored multi-line chart per multi-series group.
        for g in multi_series_groups:
            series_dict = self._series_history[g]
            keys = sorted(series_dict.keys())
            xs = [series_dict[k]["xs"] for k in keys]
            ys = [series_dict[k]["ys"] for k in keys]
            fig = _colored_multiline_plot(xs, ys, keys, title=g, xname="epoch")
            try:
                if fig is not None:
                    flat[f"{prefix}plots/{g}"] = fig
                else:
                    flat[f"{prefix}plots/{g}"] = self._wandb.plot.line_series(
                        xs=xs, ys=ys, keys=keys, title=g, xname="epoch"
                    )
            except Exception:
                pass

        try:
            self._wandb.log(flat)
        except Exception:
            pass

    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, list):
            return "[" + ",".join(
                f"{x:.3g}" if isinstance(x, float) else str(x) for x in v
            ) + "]"
        return str(v)

    def close(self):
        self.remove_hooks()
        try:
            self._jsonl_fh.close()
        except Exception:
            pass
        try:
            self._text_fh.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
