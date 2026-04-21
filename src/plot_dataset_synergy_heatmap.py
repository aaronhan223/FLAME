import csv
import os
from collections import defaultdict

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


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(project_root, "src")
    results_dir = os.path.join(src_dir, "results")
    csv_path = os.path.join(results_dir, "task_synergy.csv")
    out_png = os.path.join(results_dir, "dataset_synergy_heatmap.png")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Synergy CSV not found at {csv_path}. Run run_all_synergy.sh first."
        )

    pairs = load_synergy(csv_path)

    # Map each task to its dataset
    task_to_dataset = {
        "ihm": "MIMIC",
        "los": "MIMIC",
        "readmission": "eICU",
        "mortality": "eICU",
        "birads": "EMBED",
        "risk": "EMBED",
        "density": "EMBED",
    }

    datasets = ["MIMIC", "eICU", "EMBED"]
    ds_idx = {d: i for i, d in enumerate(datasets)}
    n = len(datasets)

    # Aggregate synergies per dataset pair
    agg = defaultdict(list)
    for a, b, val in pairs:
        da = task_to_dataset.get(a)
        db = task_to_dataset.get(b)
        if da is None or db is None:
            continue
        key = tuple(sorted((da, db)))
        agg[key].append(val)

    mat = np.full((n, n), np.nan, dtype=float)
    for (da, db), vals in agg.items():
        i, j = ds_idx[da], ds_idx[db]
        mean_val = float(np.mean(vals))
        mat[i, j] = mean_val
        mat[j, i] = mean_val

    # Color scale directly from raw dataset-level synergies
    finite_vals = mat[np.isfinite(mat)]
    if finite_vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
        if abs(vmax - vmin) < 1e-9:
            vmin, vmax = vmin - 1.0, vmax + 1.0

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_yticklabels(datasets)

    for i in range(n):
        for j in range(n):
            if np.isfinite(mat[i, j]):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Synergy (mean over task pairs)")

    ax.set_title("Dataset-level synergy", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Saved dataset-level synergy heatmap to {out_png}")


if __name__ == "__main__":
    main()

