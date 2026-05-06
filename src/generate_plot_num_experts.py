"""
Publication-quality line plots for expert ablation study.
  - X-axis: log2 scale (2, 4, 8, 16, 32 experts; 5 removed)
  - Two side-by-side panels (AUROC, AUPRC), independent x-axes
  - Bottom legend in dataset columns (one block per dataset)
  - Colors/markers from original script
  - Paper-ready: sans-serif font, tight layout, exports PDF + PNG
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Data (5-expert rows removed) ─────────────────────────────────────────────

experts = [2, 4, 8, 16, 32]

auroc = {
    "48-IHM":   [0.798, 0.808, 0.817, 0.803, 0.805],
    "LOS":      [0.816, 0.816, 0.817, 0.821, 0.804],
    "25-PHENO": [0.693, 0.687, 0.684, 0.660, 0.646],
    "MOR":      [0.839, 0.838, 0.840, 0.834, 0.838],
    "RAD":      [0.764, 0.762, 0.760, 0.750, 0.751],
    "BIRADS":   [0.796, 0.797, 0.799, 0.803, 0.785],
    "RISK":     [0.721, 0.703, 0.723, 0.729, 0.726],
    "DENSITY":  [0.914, 0.920, 0.915, 0.920, 0.908],
    "DIAG":     [0.767, 0.773, 0.761, 0.666, 0.749],
}

auprc = {
    "48-IHM":   [0.434, 0.458, 0.469, 0.446, 0.433],
    "LOS":      [0.732, 0.723, 0.741, 0.735, 0.700],
    "25-PHENO": [0.445, 0.441, 0.436, 0.407, 0.393],
    "MOR":      [0.278, 0.274, 0.277, 0.255, 0.265],
    "RAD":      [0.452, 0.451, 0.455, 0.444, 0.445],
    "BIRADS":   [0.536, 0.542, 0.551, 0.539, 0.542],
    "RISK":     [0.134, 0.141, 0.137, 0.143, 0.154],
    "DENSITY":  [0.724, 0.746, 0.734, 0.752, 0.715],
    "DIAG":     [0.646, 0.638, 0.642, 0.526, 0.642],
}

dataset_groups = {
    "MIMIC-IV": ["48-IHM", "LOS", "25-PHENO"],
    "eICU":     ["MOR", "RAD"],
    "EMBED":    ["BIRADS", "RISK", "DENSITY"],
    "ADNI":     ["DIAG"],
}

# ── Original colors/markers from reference script ─────────────────────────────
GROUP_COLORS = {
    "MIMIC-IV": "#0072B2",
    "eICU":     "#D55E00",
    "EMBED":    "#009E73",
    "ADNI":     "#CC79A7",
}

MARKERS    = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]
LINESTYLES = ["-", "--", ":"]

MIMIC_LINESTYLES = [(0, (3, 2)), (0, (3, 2, 1, 2)), (0, (2, 2))]

# ── Global rcParams ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.9,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "grid.alpha":        0.38,
})

# FS = dict(title=15, axis=13, tick=11.5, legend=11)
FS = dict(title=18, axis=15, tick=14, legend=14)


def build_task_styles() -> dict:
    task_styles = {}
    t_global = 0
    for group, tasks in dataset_groups.items():
        ls_list = MIMIC_LINESTYLES if group == "MIMIC-IV" else LINESTYLES
        for li, task in enumerate(tasks):
            task_styles[task] = {
                "color":     GROUP_COLORS[group],
                "linestyle": ls_list[li % len(ls_list)],
                "marker":    MARKERS[t_global % len(MARKERS)],
            }
            t_global += 1
    return task_styles


def grouped_legend_handles_labels(task_styles: dict):
    """One legend column per dataset (header + tasks), padded to equal row count."""
    _dummy = Line2D([], [], linestyle="none", marker="none", color="none")

    columns = []
    for group, tasks in dataset_groups.items():
        col_handles = [Line2D([0], [0], color="none")]
        col_labels = [group]
        ls_list = MIMIC_LINESTYLES if group == "MIMIC-IV" else LINESTYLES
        for li, task in enumerate(tasks):
            st = task_styles[task]
            col_handles.append(
                Line2D(
                    [0], [0],
                    color=st["color"],
                    linestyle=st["linestyle"],
                    marker=st["marker"],
                    markersize=7,
                    linewidth=1.8,
                )
            )
            col_labels.append(task)
        columns.append(list(zip(col_handles, col_labels)))

    ncols = len(columns)
    nrows = max(len(col) for col in columns)

    legend_handles, legend_labels = [], []
    for r in range(nrows):
        for col in columns:
            if r < len(col):
                h, lab = col[r]
            else:
                h, lab = _dummy, ""
            legend_handles.append(h)
            legend_labels.append(lab)

    return legend_handles, legend_labels, ncols


def plot_panel(ax, data: dict, metric: str, title: str, task_styles: dict, *, show_xlabel: bool):
    task_order = list(data.keys())

    for task in task_order:
        st = task_styles[task]
        ax.plot(
            experts,
            data[task],
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            linewidth=2.0,
            markersize=8,
            markeredgewidth=1.2,
            markeredgecolor="white",
            label=task,
            zorder=3,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(experts)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xticklabels([str(e) for e in experts], fontsize=FS["tick"])
    ax.set_xlim(1.6, 38)

    ax.tick_params(axis="y", labelsize=FS["tick"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    if show_xlabel:
        ax.set_xlabel(r"Number of Experts (log$_2$ scale)", fontsize=FS["axis"], labelpad=7)
    else:
        ax.set_xlabel("")
    ax.set_ylabel(f"Macro-avg. {metric}", fontsize=FS["axis"], labelpad=7)
    ax.set_title(title, fontsize=FS["title"], fontweight="bold", pad=9)


def make_combined_plot(out_prefix: str):
    task_styles = build_task_styles()
    legend_handles, legend_labels, legend_ncol = grouped_legend_handles_labels(task_styles)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.35),
        sharex=False,
        constrained_layout=False,
    )

    plot_panel(
        axes[0],
        auroc,
        "AUROC",
        "Expert Ablation (AUROC)",
        task_styles,
        show_xlabel=True,
    )
    plot_panel(
        axes[1],
        auprc,
        "AUPRC",
        "Expert Ablation (AUPRC)",
        task_styles,
        show_xlabel=True,
    )

    leg = fig.legend(
        legend_handles,
        legend_labels,
        fontsize=FS["legend"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        borderaxespad=0,
        framealpha=0.92,
        edgecolor="#d0d0d0",
        ncol=legend_ncol,
        handlelength=2.2,
        labelspacing=0.28,
        borderpad=0.45,
        columnspacing=1.35,
    )
    for text in leg.get_texts():
        if text.get_text() in dataset_groups:
            text.set_fontweight("bold")
    leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.12, bottom=0.17)

    fig.savefig(f"{out_prefix}.pdf", format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{out_prefix}.png", format="png", bbox_inches="tight", dpi=300)
    print(f"Saved {out_prefix}.pdf / .png")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_combined_plot("./num_expert_ablation_auroc_auprc")
