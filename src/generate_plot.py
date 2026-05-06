"""
Publication-quality line plots for expert ablation study.
Guidelines:
  - Uses seaborn/matplotlib with a clean, paper-friendly style
  - Font sizes scaled up for readability after PDF compression
  - Tight layout, vector-ready (saves as PDF + PNG)
  - Color palette is colorblind-friendly (ColorBrewer "tab10" subset)
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────

experts = [2, 4, 5, 8, 16, 32]

# AUROC
auroc = {
    "48-IHM":    [0.798, 0.808, 0.809, 0.817, 0.803, 0.805],
    "LOS":       [0.816, 0.816, 0.815, 0.817, 0.821, 0.804],
    "25-PHENO":  [0.693, 0.687, 0.688, 0.684, 0.660, 0.646],
    "MOR":       [0.839, 0.838, 0.828, 0.840, 0.834, 0.838],
    "RAD":       [0.764, 0.762, 0.759, 0.760, 0.750, 0.751],
    "BIRADS":    [0.796, 0.797, 0.784, 0.799, 0.803, 0.785],
    "RISK":      [0.721, 0.703, 0.701, 0.723, 0.729, 0.726],
    "DENSITY":   [0.914, 0.920, 0.921, 0.915, 0.920, 0.908],
    "DIAG":      [0.767, 0.773, 0.754, 0.761, 0.666, 0.749],
}

# AUPRC
auprc = {
    "48-IHM":    [0.434, 0.458, 0.455, 0.469, 0.446, 0.433],
    "LOS":       [0.732, 0.723, 0.740, 0.741, 0.735, 0.700],
    "25-PHENO":  [0.445, 0.441, 0.439, 0.436, 0.407, 0.393],
    "MOR":       [0.278, 0.274, 0.260, 0.277, 0.255, 0.265],
    "RAD":       [0.452, 0.451, 0.453, 0.455, 0.444, 0.445],
    "BIRADS":    [0.536, 0.542, 0.537, 0.551, 0.539, 0.542],
    "RISK":      [0.134, 0.141, 0.132, 0.137, 0.143, 0.154],
    "DENSITY":   [0.724, 0.746, 0.755, 0.734, 0.752, 0.715],
    "DIAG":      [0.646, 0.638, 0.635, 0.642, 0.526, 0.642],
}

# Dataset groups → used for vertical span shading
dataset_groups = {
    "MIMIC IV":  ["48-IHM", "LOS", "25-PHENO"],
    "eICU":      ["MOR", "RAD"],
    "EMBED":     ["BIRADS", "RISK", "DENSITY"],
    "ADNI":      ["DIAG"],
}

# ── Style ─────────────────────────────────────────────────────────────────────

# Colorblind-friendly palette (Wong 2011 + tab10 blend, 9 colors)
COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # purple-pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow (dark stroke)
    "#999999",  # grey
    "#000000",  # black
]

MARKERS = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]

# Global style
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    1.2,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  5,
    "ytick.major.size":  5,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.45,
})

FONT_SIZES = {
    "title":      16,
    "axis_label": 14,
    "tick":       13,
    "legend":     11,
    "annotation": 10,
}

# ── Helper ────────────────────────────────────────────────────────────────────

def _group_x_spans(task_order):
    """Return (xmin, xmax, label) spans in axes-fraction coords for shading."""
    spans = []
    n = len(task_order)
    for group, tasks in dataset_groups.items():
        idxs = [task_order.index(t) for t in tasks if t in task_order]
        if not idxs:
            continue
        spans.append((min(idxs) - 0.5, max(idxs) + 0.5, group))
    return spans


def make_plot(data: dict, metric: str, out_prefix: str):
    task_order = list(data.keys())
    n_tasks    = len(task_order)
    x          = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(12, 5))

    # ── Alternating group shading ──────────────────────────────────────────
    shade_colors = ["#f0f4ff", "#fff8f0", "#f0fff4", "#fdf0ff"]
    for i, (group, tasks) in enumerate(dataset_groups.items()):
        idxs = [task_order.index(t) for t in tasks if t in task_order]
        if not idxs:
            continue
        xlo = min(idxs) - 0.5
        xhi = max(idxs) + 0.5
        ax.axvspan(xlo, xhi, color=shade_colors[i % len(shade_colors)],
                   alpha=0.55, zorder=0)
        ax.text((xlo + xhi) / 2, 1.01, group,
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom",
                fontsize=FONT_SIZES["annotation"],
                fontstyle="italic", color="#555555")

    # ── Lines ──────────────────────────────────────────────────────────────
    for idx, (expert_count, values) in enumerate(
            sorted(data.items() if isinstance(data, dict) else [])):
        pass  # placeholder; iterate over experts below

    # data[task] = list of values across experts → transpose for plotting
    # Plot one line per expert configuration
    expert_labels = [str(e) for e in experts]

    for i, (n_exp, label) in enumerate(zip(experts, expert_labels)):
        y = [data[task][i] for task in task_order]
        ax.plot(x, y,
                color=COLORS[i % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                linewidth=2.0,
                markersize=7,
                markeredgewidth=1.2,
                markeredgecolor="white",
                label=f"{n_exp} experts",
                zorder=3)

    # ── Axes formatting ────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(task_order, fontsize=FONT_SIZES["tick"],
                       rotation=20, ha="right")
    ax.tick_params(axis="y", labelsize=FONT_SIZES["tick"])

    ax.set_ylabel(f"Macro-averaged {metric}", fontsize=FONT_SIZES["axis_label"],
                  labelpad=8)
    ax.set_title(f"Expert Ablation — {metric} across Tasks",
                 fontsize=FONT_SIZES["title"], pad=14, fontweight="bold")

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xlim(-0.5, n_tasks - 0.5)

    # ── Legend ─────────────────────────────────────────────────────────────
    leg = ax.legend(title="# Experts", title_fontsize=FONT_SIZES["legend"],
                    fontsize=FONT_SIZES["legend"],
                    loc="lower left", framealpha=0.9,
                    edgecolor="#cccccc", ncol=3)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()

    pdf_path = f"{out_prefix}.pdf"
    png_path = f"{out_prefix}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    print(f"Saved: {pdf_path}  |  {png_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    make_plot(auroc, "AUROC", "./expert_ablation_auroc")
    make_plot(auprc, "AUPRC", "./expert_ablation_auprc")