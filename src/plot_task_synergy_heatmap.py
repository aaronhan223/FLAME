import csv
import os

import numpy as np
import matplotlib.pyplot as plt


def load_synergy(csv_path):
    pairs = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                val = float(row["synergy"])
            except (KeyError, ValueError):
                continue
            pairs.append((row["task_a"], row["task_b"], val))
    return pairs


def build_matrix(pairs, tasks):
    idx = {t: i for i, t in enumerate(tasks)}
    n = len(tasks)
    mat = np.full((n, n), np.nan, dtype=float)
    for a, b, val in pairs:
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        mat[i, j] = val
        mat[j, i] = val  # assume symmetry
    return mat


def plot_heatmap(mat_raw, tasks, out_path):
    n = len(tasks)

    # Determine color scale directly from raw values
    finite_vals = mat_raw[np.isfinite(mat_raw)]
    if finite_vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
        if abs(vmax - vmin) < 1e-9:
            # Avoid zero range
            vmin, vmax = vmin - 1.0, vmax + 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat_raw, cmap="Blues", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_yticklabels(tasks)

    # Annotate with raw synergy values, to two decimals.
    for i in range(n):
        for j in range(n):
            if np.isfinite(mat_raw[i, j]):
                ax.text(
                    j,
                    i,
                    f"{mat_raw[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Synergy")

    ax.set_title("Task-pair synergy", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(project_root, "src")
    results_dir = os.path.join(src_dir, "results")
    csv_path = os.path.join(results_dir, "task_synergy.csv")
    out_png = os.path.join(results_dir, "task_synergy_heatmap.png")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Synergy CSV not found at {csv_path}. Run run_all_synergy.sh first."
        )

    pairs = load_synergy(csv_path)

    # Global task order – adjust if you change which tasks you use.
    tasks = ["ihm", "los", "readmission", "mortality", "birads", "risk", "density"]

    mat_raw = build_matrix(pairs, tasks)
    plot_heatmap(mat_raw, tasks, out_png)

    print(f"Saved task-pair synergy heatmap to {out_png}")


if __name__ == "__main__":
    main()

