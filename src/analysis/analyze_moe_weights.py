"""
Analyze and compare MoE expert/router weights between IHM base and LOS transfer models.

Analyses performed:
  1. Weight-space cosine similarity (expert-to-expert, within and across models)
  2. L2 distance: how much each expert changed from IHM → LOS
  3. CKA (linear): representational similarity between experts
  4. PCA visualization of flattened expert weight vectors
  5. Router analysis: gate weight similarity, temporal pool drift
  6. Weight Watcher spectral analysis (alpha metric per expert)

Usage:
    python -m src.analysis.analyze_moe_weights \
        --ihm_model_path checkpoints/flame/multitask/ihm/ihm_TS-Text_mod_drop_rate_0.0.pt \
        --los_model_path checkpoints/flame/multitask/ihm/TS-Text/los_TS-Text-CXR_transfer_moe_from_ihm.pt \
        --output_dir analysis_results/moe_comparison
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(1, os.getcwd())

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from itertools import combinations
import warnings


# ─── Helper: extract MoE components from a saved model ──────────────────────

def extract_moe_components(model):
    """Extract expert and router state dicts from all transformer layers.

    Returns:
        experts: dict mapping (layer_idx, expert_idx) → OrderedDict of params
        routers: dict mapping (layer_idx, modality_name) → OrderedDict of params
    """
    if hasattr(model, 'state_dict'):
        sd = model.state_dict()
    else:
        sd = model

    experts = {}
    routers = {}

    for key, val in sd.items():
        # Expert keys: ...layers.{L}.moe.experts.{E}.{param}
        if '.moe.experts.' in key:
            parts = key.split('.')
            layer_idx = int(parts[parts.index('layers') + 1])
            expert_idx = int(parts[parts.index('experts') + 1])
            param_name = '.'.join(parts[parts.index('experts') + 2:])
            k = (layer_idx, expert_idx)
            if k not in experts:
                experts[k] = OrderedDict()
            experts[k][param_name] = val.cpu()

        # Router keys: ...layers.{L}.moe.routers.{mod}.{param}
        if '.moe.routers.' in key:
            parts = key.split('.')
            layer_idx = int(parts[parts.index('layers') + 1])
            mod_idx = parts.index('routers') + 1
            mod_name = parts[mod_idx]
            param_name = '.'.join(parts[mod_idx + 1:])
            k = (layer_idx, mod_name)
            if k not in routers:
                routers[k] = OrderedDict()
            routers[k][param_name] = val.cpu()

    return experts, routers


def flatten_params(param_dict):
    """Flatten an OrderedDict of tensors into a single 1D vector."""
    return torch.cat([p.flatten().float() for p in param_dict.values()])


# ─── 1. Cosine Similarity ───────────────────────────────────────────────────

def compute_cosine_similarity_matrix(expert_vectors):
    """Compute pairwise cosine similarity matrix for a list of vectors.

    Args:
        expert_vectors: dict mapping label → 1D tensor
    Returns:
        labels: list of str
        sim_matrix: np.ndarray [N, N]
    """
    labels = list(expert_vectors.keys())
    vecs = torch.stack([expert_vectors[l] for l in labels])
    vecs_norm = vecs / vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sim = (vecs_norm @ vecs_norm.T).numpy()
    return labels, sim


def plot_similarity_matrix(labels, sim_matrix, title, save_path):
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), max(6, len(labels) * 0.6)))
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center', fontsize=7)

    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ─── 2. L2 Distance (IHM → LOS drift per expert) ───────────────────────────

def compute_expert_drift(ihm_experts, los_experts):
    """Compute L2 distance between corresponding experts across models.

    Returns:
        drift: dict mapping (layer, expert) → float (L2 distance)
        relative_drift: dict mapping (layer, expert) → float (L2 / ||ihm||)
    """
    drift = {}
    relative_drift = {}
    common_keys = set(ihm_experts.keys()) & set(los_experts.keys())

    for k in sorted(common_keys):
        v_ihm = flatten_params(ihm_experts[k])
        v_los = flatten_params(los_experts[k])
        d = (v_ihm - v_los).norm().item()
        ihm_norm = v_ihm.norm().item()
        drift[k] = d
        relative_drift[k] = d / ihm_norm if ihm_norm > 0 else float('inf')

    return drift, relative_drift


# ─── 3. CKA (Linear) ────────────────────────────────────────────────────────

def linear_CKA(X, Y):
    """Compute linear CKA between two weight matrices.

    Args:
        X, Y: 2D tensors [samples, features] or flattened weight matrices
              reshaped to [output_dim, input_dim].
    Returns:
        float: CKA similarity in [0, 1].
    """
    X = X.float()
    Y = Y.float()

    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y

    hsic_xy = (XtY * XtY).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy).clamp(min=1e-10)
    return (hsic_xy / denom).item()


def compute_cka_matrix(expert_params, key_name='fc1.weight'):
    """Compute pairwise CKA between experts using a specific weight matrix.

    Args:
        expert_params: dict mapping label → OrderedDict of params
        key_name: which parameter to use (e.g., 'fc1.weight')
    Returns:
        labels, cka_matrix
    """
    labels = []
    matrices = []
    for label, params in sorted(expert_params.items()):
        if key_name in params:
            labels.append(str(label))
            matrices.append(params[key_name].float())

    n = len(labels)
    cka_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cka_mat[i, j] = linear_CKA(matrices[i], matrices[j])

    return labels, cka_mat


# ─── 4. PCA Visualization ───────────────────────────────────────────────────

def plot_pca_experts(ihm_experts, los_experts, save_path):
    """PCA on flattened expert weight vectors, colored by model and expert."""
    all_vectors = []
    labels = []
    colors = []
    markers = []

    for (layer, eidx), params in sorted(ihm_experts.items()):
        all_vectors.append(flatten_params(params).numpy())
        labels.append(f'IHM L{layer} E{eidx}')
        colors.append(f'C{eidx}')
        markers.append('o')

    for (layer, eidx), params in sorted(los_experts.items()):
        all_vectors.append(flatten_params(params).numpy())
        labels.append(f'LOS L{layer} E{eidx}')
        colors.append(f'C{eidx}')
        markers.append('^')

    X = np.stack(all_vectors)
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    coords = X_centered @ Vt[:2].T
    explained = (S[:2] ** 2) / (S ** 2).sum() * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (label, c, m) in enumerate(zip(labels, colors, markers)):
        ax.scatter(coords[i, 0], coords[i, 1], c=c, marker=m, s=100,
                   edgecolors='black', linewidths=0.5)
        ax.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=7,
                    xytext=(5, 5), textcoords='offset points')

    # Draw arrows from IHM → LOS for same (layer, expert)
    ihm_map = {}
    for i, l in enumerate(labels):
        if l.startswith('IHM'):
            ihm_map[l.replace('IHM ', '')] = i
    for i, l in enumerate(labels):
        if l.startswith('LOS'):
            key = l.replace('LOS ', '')
            if key in ihm_map:
                j = ihm_map[key]
                ax.annotate('', xy=(coords[i, 0], coords[i, 1]),
                            xytext=(coords[j, 0], coords[j, 1]),
                            arrowprops=dict(arrowstyle='->', color='gray',
                                            lw=1, alpha=0.5))

    ax.set_xlabel(f'PC1 ({explained[0]:.1f}% var)')
    ax.set_ylabel(f'PC2 ({explained[1]:.1f}% var)')
    ax.set_title('PCA of Expert Weight Vectors (o=IHM, ^=LOS)')
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='gray', label='IHM', linestyle='None', markersize=8),
        plt.Line2D([0], [0], marker='^', color='gray', label='LOS', linestyle='None', markersize=8),
    ])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ─── 5. Router Analysis ─────────────────────────────────────────────────────

def analyze_routers(ihm_routers, los_routers, output_dir):
    """Compare router gate weights and temporal pool between models."""
    print("\n=== Router Analysis ===")

    # Gate weight cosine similarity
    gate_vectors = {}
    for (layer, mod), params in sorted(ihm_routers.items()):
        if 'w_gate' in params:
            gate_vectors[f'IHM L{layer} {mod}'] = params['w_gate'].flatten()
    for (layer, mod), params in sorted(los_routers.items()):
        if 'w_gate' in params:
            gate_vectors[f'LOS L{layer} {mod}'] = params['w_gate'].flatten()

    if gate_vectors:
        labels, sim = compute_cosine_similarity_matrix(gate_vectors)
        plot_similarity_matrix(labels, sim, 'Router Gate Weight (w_gate) Cosine Similarity',
                               os.path.join(output_dir, 'router_gate_cosine_sim.png'))

    # Temporal pool drift (should be ~0 if frozen correctly)
    print("\n  Temporal pool drift (should be ~0 if frozen):")
    common = set(ihm_routers.keys()) & set(los_routers.keys())
    for k in sorted(common):
        if 'temporal_pool.query' in ihm_routers[k] and 'temporal_pool.query' in los_routers[k]:
            drift = (ihm_routers[k]['temporal_pool.query'] - los_routers[k]['temporal_pool.query']).norm().item()
            print(f"    {k}: L2 drift = {drift:.6f}")


# ─── 6. Weight Watcher Spectral Analysis ────────────────────────────────────

def spectral_analysis(experts, model_name, output_dir):
    """Compute effective rank and spectral decay (alpha) for each expert's fc1 weight.

    Alpha < 2: heavy-tailed, potentially well-trained
    Alpha 2-4: moderate, typical
    Alpha > 4: light-tailed, possibly undertrained
    """
    print(f"\n=== Spectral Analysis ({model_name}) ===")
    results = {}
    for (layer, eidx), params in sorted(experts.items()):
        for pname in ['fc1.weight', 'fc2.weight', 'temporal_conv.weight']:
            if pname not in params:
                continue
            W = params[pname].float()
            if W.dim() == 3:  # Conv1d: [out, in, kernel]
                W = W.reshape(W.shape[0], -1)

            S = torch.linalg.svdvals(W)
            S = S[S > 1e-8]

            # Effective rank (entropy-based)
            p = S / S.sum()
            eff_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-10))).item()

            # Power-law alpha via log-log regression on eigenvalues
            evals = (S ** 2).numpy()
            log_evals = np.log(evals[evals > 0])
            log_ranks = np.log(np.arange(1, len(log_evals) + 1))
            if len(log_evals) > 2:
                alpha = -np.polyfit(log_ranks, log_evals, 1)[0]
            else:
                alpha = float('nan')

            results[(layer, eidx, pname)] = {
                'eff_rank': eff_rank,
                'alpha': alpha,
                'max_sv': S[0].item(),
                'cond_number': (S[0] / S[-1]).item() if len(S) > 1 else float('inf'),
            }
            print(f"  L{layer} E{eidx} {pname:20s}: alpha={alpha:.2f}, eff_rank={eff_rank:.1f}, "
                  f"cond={results[(layer, eidx, pname)]['cond_number']:.1f}")

    return results


# ─── 7. Per-layer expert weight comparison (bar chart) ──────────────────────

def plot_expert_drift_per_layer(drift, relative_drift, save_path):
    """Bar chart of L2 drift per expert, grouped by layer."""
    if not drift:
        return

    layers = sorted(set(l for l, _ in drift.keys()))
    expert_ids = sorted(set(e for _, e in drift.keys()))
    n_layers = len(layers)
    n_experts = len(expert_ids)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute drift
    x = np.arange(n_layers)
    width = 0.8 / n_experts
    for i, eidx in enumerate(expert_ids):
        vals = [drift.get((l, eidx), 0) for l in layers]
        axes[0].bar(x + i * width, vals, width, label=f'Expert {eidx}')
    axes[0].set_xticks(x + width * (n_experts - 1) / 2)
    axes[0].set_xticklabels([f'Layer {l}' for l in layers])
    axes[0].set_ylabel('L2 Distance')
    axes[0].set_title('Absolute Expert Weight Drift (IHM → LOS)')
    axes[0].legend()

    # Relative drift
    for i, eidx in enumerate(expert_ids):
        vals = [relative_drift.get((l, eidx), 0) for l in layers]
        axes[1].bar(x + i * width, vals, width, label=f'Expert {eidx}')
    axes[1].set_xticks(x + width * (n_experts - 1) / 2)
    axes[1].set_xticklabels([f'Layer {l}' for l in layers])
    axes[1].set_ylabel('Relative Drift (L2 / ||IHM||)')
    axes[1].set_title('Relative Expert Weight Drift (IHM → LOS)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ─── 8. SVD Layerwise Plots ──────────────────────────────────────────────────

def plot_svd_experts(ihm_experts, los_experts, svd_dir):
    """Plot singular value spectra for each expert's weight matrices, per layer.

    For each layer, produces side-by-side IHM vs LOS subplots for each weight
    type (fc1, fc2, temporal_conv), with one curve per expert.
    """
    os.makedirs(svd_dir, exist_ok=True)
    layers = sorted(set(l for l, _ in ihm_experts.keys()))
    expert_ids = sorted(set(e for _, e in ihm_experts.keys()))
    weight_names = ['fc1.weight', 'fc2.weight', 'temporal_conv.weight']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for weight_name in weight_names:
        for layer in layers:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

            for ax, (model_name, experts) in zip(axes, [('IHM', ihm_experts), ('LOS', los_experts)]):
                for eidx in expert_ids:
                    key = (layer, eidx)
                    if key not in experts or weight_name not in experts[key]:
                        continue
                    W = experts[key][weight_name].float()
                    if W.dim() == 3:  # Conv1d: [out, in, kernel] → [out, in*kernel]
                        W = W.reshape(W.shape[0], -1)
                    S = torch.linalg.svdvals(W).numpy()

                    ax.semilogy(np.arange(1, len(S) + 1), S,
                                color=colors[eidx % len(colors)],
                                linewidth=2, label=f'Expert {eidx}',
                                marker='o', markersize=3, alpha=0.8)

                ax.set_xlabel('Singular Value Index')
                ax.set_ylabel('Singular Value (log scale)')
                ax.set_title(f'{model_name} — Layer {layer}')
                ax.legend()
                ax.grid(True, alpha=0.3)

            fig.suptitle(f'SVD Spectrum (raw): {weight_name} — Layer {layer}', fontsize=13, fontweight='bold')
            plt.tight_layout()
            save_path = os.path.join(svd_dir, f'svd_{weight_name.replace(".", "_")}_layer{layer}.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"  Saved: {save_path}")

    # Normalized version: each curve divided by its top singular value
    for weight_name in weight_names:
        for layer in layers:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

            for ax, (model_name, experts) in zip(axes, [('IHM', ihm_experts), ('LOS', los_experts)]):
                for eidx in expert_ids:
                    key = (layer, eidx)
                    if key not in experts or weight_name not in experts[key]:
                        continue
                    W = experts[key][weight_name].float()
                    if W.dim() == 3:
                        W = W.reshape(W.shape[0], -1)
                    S = torch.linalg.svdvals(W).numpy()
                    S_norm = S / (S[0] + 1e-10)  # normalize by top SV

                    ax.semilogy(np.arange(1, len(S_norm) + 1), S_norm,
                                color=colors[eidx % len(colors)],
                                linewidth=2, label=f'Expert {eidx}',
                                marker='o', markersize=3, alpha=0.8)

                ax.set_xlabel('Singular Value Index')
                ax.set_ylabel('σ_i / σ_1 (log scale)')
                ax.set_title(f'{model_name} — Layer {layer}')
                ax.legend()
                ax.grid(True, alpha=0.3)

            fig.suptitle(f'SVD Spectrum (normalized by σ₁): {weight_name} — Layer {layer}', fontsize=13, fontweight='bold')
            plt.tight_layout()
            save_path = os.path.join(svd_dir, f'svd_norm_{weight_name.replace(".", "_")}_layer{layer}.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"  Saved: {save_path}")

    # Also plot IHM vs LOS overlay for the SAME expert (to see drift visually)
    for weight_name in weight_names:
        for layer in layers:
            for eidx in expert_ids:
                ihm_key = (layer, eidx)
                los_key = (layer, eidx)
                if ihm_key not in ihm_experts or los_key not in los_experts:
                    continue
                if weight_name not in ihm_experts[ihm_key] or weight_name not in los_experts[los_key]:
                    continue

                W_ihm = ihm_experts[ihm_key][weight_name].float()
                W_los = los_experts[los_key][weight_name].float()
                if W_ihm.dim() == 3:
                    W_ihm = W_ihm.reshape(W_ihm.shape[0], -1)
                if W_los.dim() == 3:
                    W_los = W_los.reshape(W_los.shape[0], -1)

                S_ihm = torch.linalg.svdvals(W_ihm).numpy()
                S_los = torch.linalg.svdvals(W_los).numpy()

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.semilogy(np.arange(1, len(S_ihm) + 1), S_ihm,
                            'b-o', markersize=3, linewidth=2, alpha=0.8, label='IHM')
                ax.semilogy(np.arange(1, len(S_los) + 1), S_los,
                            'r-^', markersize=3, linewidth=2, alpha=0.8, label='LOS')

                # Shade the gap
                min_len = min(len(S_ihm), len(S_los))
                x = np.arange(1, min_len + 1)
                ax.fill_between(x, S_ihm[:min_len], S_los[:min_len],
                                alpha=0.15, color='gray')

                ax.set_xlabel('Singular Value Index')
                ax.set_ylabel('Singular Value (log scale)')
                ax.set_title(f'SVD Drift: {weight_name} — Layer {layer}, Expert {eidx}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                save_path = os.path.join(svd_dir,
                    f'svd_drift_{weight_name.replace(".", "_")}_L{layer}_E{eidx}.png')
                plt.savefig(save_path, dpi=150)
                plt.close()

    print(f"  SVD drift overlays saved to {svd_dir}/")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Analyze MoE expert/router weights')
    parser.add_argument('--ihm_model_path', type=str, required=True,
                        help='Path to IHM base model checkpoint (.pt)')
    parser.add_argument('--los_model_path', type=str, required=True,
                        help='Path to LOS transfer model checkpoint (.pt)')
    parser.add_argument('--output_dir', type=str, default='analysis_results/moe_comparison',
                        help='Directory to save plots and results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output dir: {args.output_dir}")

    # Load models
    print(f"\nLoading IHM model: {args.ihm_model_path}")
    ihm_model = torch.load(args.ihm_model_path, map_location='cpu')
    print(f"Loading LOS model: {args.los_model_path}")
    los_model = torch.load(args.los_model_path, map_location='cpu')

    # Extract components
    ihm_experts, ihm_routers = extract_moe_components(ihm_model)
    los_experts, los_routers = extract_moe_components(los_model)

    print(f"\nIHM experts: {sorted(ihm_experts.keys())}")
    print(f"LOS experts: {sorted(los_experts.keys())}")
    print(f"IHM routers: {sorted(ihm_routers.keys())}")
    print(f"LOS routers: {sorted(los_routers.keys())}")

    # --- 1. Cosine similarity: all experts (within + across models) ---
    print("\n=== Cosine Similarity (Expert Weight Vectors) ===")
    all_expert_vectors = {}
    for (layer, eidx), params in sorted(ihm_experts.items()):
        all_expert_vectors[f'IHM L{layer} E{eidx}'] = flatten_params(params)
    for (layer, eidx), params in sorted(los_experts.items()):
        all_expert_vectors[f'LOS L{layer} E{eidx}'] = flatten_params(params)

    labels, sim = compute_cosine_similarity_matrix(all_expert_vectors)
    plot_similarity_matrix(labels, sim, 'Expert Weight Vector Cosine Similarity',
                           os.path.join(args.output_dir, 'expert_cosine_similarity.png'))

    # Per-layer cosine similarity
    layers = sorted(set(l for l, _ in ihm_experts.keys()))
    for layer in layers:
        layer_vectors = {}
        for (l, e), p in ihm_experts.items():
            if l == layer:
                layer_vectors[f'IHM E{e}'] = flatten_params(p)
        for (l, e), p in los_experts.items():
            if l == layer:
                layer_vectors[f'LOS E{e}'] = flatten_params(p)
        if layer_vectors:
            lbl, s = compute_cosine_similarity_matrix(layer_vectors)
            plot_similarity_matrix(lbl, s, f'Layer {layer}: Expert Cosine Similarity',
                                   os.path.join(args.output_dir, f'expert_cosine_sim_layer{layer}.png'))

    # --- 2. L2 drift ---
    print("\n=== Expert Weight Drift (IHM → LOS) ===")
    drift, rel_drift = compute_expert_drift(ihm_experts, los_experts)
    for k in sorted(drift.keys()):
        print(f"  L{k[0]} E{k[1]}: L2={drift[k]:.4f}, relative={rel_drift[k]:.4f}")
    plot_expert_drift_per_layer(drift, rel_drift,
                                os.path.join(args.output_dir, 'expert_drift.png'))

    # --- 3. CKA ---
    print("\n=== CKA (fc1.weight) ===")
    all_expert_params = {}
    for k, v in ihm_experts.items():
        all_expert_params[f'IHM L{k[0]} E{k[1]}'] = v
    for k, v in los_experts.items():
        all_expert_params[f'LOS L{k[0]} E{k[1]}'] = v

    for weight_name in ['fc1.weight', 'fc2.weight']:
        cka_labels, cka_mat = compute_cka_matrix(all_expert_params, key_name=weight_name)
        if cka_labels:
            plot_similarity_matrix(cka_labels, cka_mat,
                                   f'CKA Similarity ({weight_name})',
                                   os.path.join(args.output_dir, f'cka_{weight_name.replace(".", "_")}.png'))

    # --- 4. PCA ---
    print("\n=== PCA Visualization ===")
    plot_pca_experts(ihm_experts, los_experts,
                     os.path.join(args.output_dir, 'pca_experts.png'))

    # --- 5. Router analysis ---
    analyze_routers(ihm_routers, los_routers, args.output_dir)

    # --- 6. Spectral analysis ---
    ihm_spectral = spectral_analysis(ihm_experts, 'IHM', args.output_dir)
    los_spectral = spectral_analysis(los_experts, 'LOS', args.output_dir)

    # --- 7. SVD layerwise plots ---
    print("\n=== SVD Layerwise Plots ===")
    svd_dir = os.path.join(args.output_dir, 'SVD_plots')
    plot_svd_experts(ihm_experts, los_experts, svd_dir)

    # --- Summary: merge candidates ---
    print("\n" + "=" * 60)
    print("=== MERGE CANDIDATES (high cosine sim between experts) ===")
    print("=" * 60)
    # Within each model, per layer
    for model_name, experts in [('IHM', ihm_experts), ('LOS', los_experts)]:
        for layer in layers:
            layer_experts = {e: flatten_params(p) for (l, e), p in experts.items() if l == layer}
            eids = sorted(layer_experts.keys())
            for i, j in combinations(eids, 2):
                cos = torch.nn.functional.cosine_similarity(
                    layer_experts[i].unsqueeze(0),
                    layer_experts[j].unsqueeze(0)
                ).item()
                if cos > 0.8:
                    print(f"  {model_name} Layer {layer}: Expert {i} ↔ Expert {j} "
                          f"cosine={cos:.3f} *** HIGH - merge candidate")
                else:
                    print(f"  {model_name} Layer {layer}: Expert {i} ↔ Expert {j} "
                          f"cosine={cos:.3f}")

    # Across models, same expert
    print("\n  Cross-model (same expert, IHM vs LOS):")
    for k in sorted(set(ihm_experts.keys()) & set(los_experts.keys())):
        cos = torch.nn.functional.cosine_similarity(
            flatten_params(ihm_experts[k]).unsqueeze(0),
            flatten_params(los_experts[k]).unsqueeze(0)
        ).item()
        print(f"  L{k[0]} E{k[1]}: cosine={cos:.3f}" +
              (" (barely changed)" if cos > 0.95 else ""))

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
