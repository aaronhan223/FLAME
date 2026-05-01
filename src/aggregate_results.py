"""Aggregate `best_model_results_*.txt` across seed subdirectories and emit LaTeX tables.

Expected layout (parent dir per modality_drop_rate):

    {result_dir}/
        {seed_a}/
            best_model_results_*.txt
        {seed_b}/
            best_model_results_*.txt
        ...

For each seed, we parse the LAST `## Final Best Model Test ##` block (file is opened
with append mode during training, so multiple blocks may exist; we take the most recent
one). We aggregate per-task scalar metrics across seeds and write:

    {result_dir}/aggregated_table.tex       compact 4-col table (AUROC, AUPRC, F1, Acc)
    {result_dir}/aggregated_table_full.tex  one table per task, all metrics
    {result_dir}/aggregated_summary.csv     mean/std/n per (task, metric)
    {result_dir}/aggregated_per_seed.csv    raw per-seed values

Numbers are rounded to `--decimals` places (default 3). Standard deviation uses
``ddof=0`` (population std); pass ``--ddof 1`` for sample std.
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np


TASK_TYPES = {
    'ihm': 'binary',
    'los': 'binary',
    'pheno': 'multilabel',
    'mor': 'binary',
    'mortality': 'binary',
    'rad': 'binary',
    'readmission': 'binary',
    'birads': 'multiclass',
    'risk': 'binary',
    'density': 'multiclass',
}

DEFAULT_METRICS_BY_TYPE = {
    'binary':     ['auc',           'auprc',           'f1',       'accuracy'],
    'multilabel': ['ave_auc_macro', 'ave_auprc_macro', 'macro_f1', 'hamming_accuracy'],
    'multiclass': ['ave_auc_macro', 'ave_auprc_macro', 'macro_f1', 'accuracy'],
}

COMPACT_HEADERS = ['AUROC', 'AUPRC', 'F1', 'Accuracy']

DISPLAY_NAMES = {
    'auc': 'AUROC', 'auprc': 'AUPRC', 'f1': 'F1', 'accuracy': 'Accuracy',
    'ave_auc_micro': 'AUROC (micro)',
    'ave_auc_macro': 'AUROC (macro)',
    'ave_auc_weighted': 'AUROC (weighted)',
    'ave_auprc_micro': 'AUPRC (micro)',
    'ave_auprc_macro': 'AUPRC (macro)',
    'ave_auprc_weighted': 'AUPRC (weighted)',
    'micro_f1': 'F1 (micro)',
    'macro_f1': 'F1 (macro)',
    'weighted_f1': 'F1 (weighted)',
    'subset_accuracy': 'Accuracy (Subset)',
    'hamming_accuracy': 'Accuracy (Hamming)',
}

# Preferred display order (metrics not listed here get appended alphabetically).
METRIC_ORDER = [
    'auc',
    'ave_auc_micro', 'ave_auc_macro', 'ave_auc_weighted',
    'auprc',
    'ave_auprc_micro', 'ave_auprc_macro', 'ave_auprc_weighted',
    'f1',
    'micro_f1', 'macro_f1', 'weighted_f1',
    'accuracy', 'subset_accuracy', 'hamming_accuracy',
]


def metric_sort_key(m):
    if m in METRIC_ORDER:
        return (0, METRIC_ORDER.index(m), m)
    return (1, 0, m)


def parse_best_block(filepath):
    """Return dict[task_idx] -> dict[metric] -> float for the LAST 'Final Best Model Test'
    block in the file, or None if no such block was found."""
    with open(filepath, 'r') as fh:
        content = fh.read()

    header_re = re.compile(r'#{2,}\s*Final Best Model Test\s*#{2,}')
    parts = header_re.split(content)
    if len(parts) < 2:
        return None
    last = parts[-1]
    next_header = re.search(r'\n#{2,}', last)
    if next_header:
        last = last[:next_header.start()]

    task_re = re.compile(r'-{3,}\s*Task\s+(\d+)\s*-{3,}')
    task_parts = task_re.split(last)
    per_task = {}
    for i in range(1, len(task_parts), 2):
        task_idx = int(task_parts[i])
        task_body = task_parts[i + 1]
        per_task[task_idx] = parse_metric_lines(task_body)
    return per_task


def parse_metric_lines(text):
    metrics = {}
    for line in text.split('\n'):
        s = line.strip()
        if not s or ':' not in s:
            continue
        key, _, val = s.partition(':')
        key = key.strip()
        val = val.strip()
        if not val or val.startswith('[') or val.lower() == 'none':
            continue
        try:
            metrics[key] = float(val)
        except ValueError:
            continue
    return metrics


def task_names_from_dir(result_dir):
    """Infer task names from a path like .../{task_spec}/mod_drop_rate_X/."""
    rd = os.path.abspath(result_dir).rstrip(os.sep)
    parent = os.path.dirname(rd)
    task_spec = os.path.basename(parent)
    return task_spec.split('-')


def find_seed_dirs(result_dir):
    out = []
    for entry in os.listdir(result_dir):
        full = os.path.join(result_dir, entry)
        if not os.path.isdir(full):
            continue
        if glob.glob(os.path.join(full, 'best_model_results_*.txt')):
            out.append((entry, full))
    try:
        out.sort(key=lambda x: int(x[0]))
    except ValueError:
        out.sort(key=lambda x: x[0])
    return out


def aggregate(per_seed, ddof=0):
    """per_seed: list of (seed, dict[task_idx][metric] -> float)
    Returns: dict[task_idx][metric] -> (mean, std, n, values)"""
    agg = defaultdict(lambda: defaultdict(list))
    for _seed, task_metrics in per_seed:
        for task_idx, metrics in task_metrics.items():
            for k, v in metrics.items():
                agg[task_idx][k].append(v)
    out = {}
    for task_idx, metrics in agg.items():
        out[task_idx] = {}
        for k, vs in metrics.items():
            mean = float(np.mean(vs))
            if len(vs) > max(1, ddof):
                std = float(np.std(vs, ddof=ddof))
            else:
                std = 0.0
            out[task_idx][k] = (mean, std, len(vs), vs)
    return out


def fmt_cell(mean, std, decimals):
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def task_type(task_name):
    return TASK_TYPES.get(task_name.lower(), 'binary')


def make_compact_table(agg, task_names, n_seeds, decimals):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{l|" + "c" * len(COMPACT_HEADERS) + "}",
        r"\hline",
        "Task & " + " & ".join(COMPACT_HEADERS) + r" \\",
        r"\hline",
    ]
    for ii in sorted(agg.keys()):
        tname = task_names[ii] if ii < len(task_names) else f"task{ii}"
        ttype = task_type(tname)
        wanted = DEFAULT_METRICS_BY_TYPE[ttype]
        cells = []
        for m in wanted:
            if m in agg[ii]:
                mean, std, _, _ = agg[ii][m]
                cells.append(fmt_cell(mean, std, decimals))
            else:
                cells.append("--")
        lines.append(f"{tname.upper()} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{Mean $\pm$ std across " + f"{n_seeds}" + r" seeds (binary tasks: AUROC/AUPRC/F1/Accuracy; multilabel: macro AUROC/AUPRC, macro F1, Hamming accuracy; multiclass: macro AUROC/AUPRC, macro F1, top-1 accuracy).}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def make_full_table(agg, task_names, n_seeds, decimals):
    sections = []
    for ii in sorted(agg.keys()):
        tname = task_names[ii] if ii < len(task_names) else f"task{ii}"
        ttype = task_type(tname)
        metrics_list = sorted(agg[ii].keys())
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{l|c}",
            r"\hline",
            r"Metric & Mean $\pm$ Std \\",
            r"\hline",
        ]
        for m in metrics_list:
            mean, std, _, _ = agg[ii][m]
            disp = DISPLAY_NAMES.get(m, m.replace('_', r'\_'))
            lines.append(f"{disp} & {fmt_cell(mean, std, decimals)} " + r"\\")
        lines += [
            r"\hline",
            r"\end{tabular}",
            r"\caption{" + f"{tname.upper()} ({ttype}); mean $\\pm$ std across {n_seeds} seeds." + r"}",
            r"\end{table}",
        ]
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def make_per_seed_table(per_seed, agg, task_names, decimals):
    """One table: rows grouped by metric (major) then task (sub), cols = each seed +
    final mean $\\pm$ std."""
    seeds = [s for s, _ in per_seed]
    seed_lookup = {s: tm for s, tm in per_seed}
    col_spec = "ll|" + "c" * len(seeds) + "|c"
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline",
        "Metric & Task & " + " & ".join(f"Seed {s}" for s in seeds) + r" & Mean $\pm$ Std \\",
        r"\hline",
    ]
    all_metrics = set()
    for ti in agg:
        all_metrics.update(agg[ti].keys())
    metrics_sorted = sorted(all_metrics, key=metric_sort_key)
    task_indices = sorted(agg.keys())

    for m in metrics_sorted:
        disp = DISPLAY_NAMES.get(m, m.replace('_', r'\_'))
        rows = []
        for ti in task_indices:
            if m not in agg[ti]:
                continue
            tname = task_names[ti] if ti < len(task_names) else f"task{ti}"
            cells = []
            for s in seeds:
                v = seed_lookup.get(s, {}).get(ti, {}).get(m)
                cells.append(f"{v:.{decimals}f}" if v is not None else "--")
            mean, std, _, _ = agg[ti][m]
            cells.append(f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$")
            rows.append((tname.upper(), cells))
        if not rows:
            continue
        for j, (tname, cells) in enumerate(rows):
            row_metric = disp if j == 0 else ""
            lines.append(f"{row_metric} & {tname} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines += [
        r"\end{tabular}",
        r"\caption{Per-seed test performance grouped by metric (with task as sub-row); final column shows mean $\pm$ std across seeds.}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def write_per_seed_csv(per_seed, task_names, out_path):
    seeds = [s for s, _ in per_seed]
    all_metrics, all_tasks = set(), set()
    for _, task_metrics in per_seed:
        for tidx, m in task_metrics.items():
            all_tasks.add(tidx)
            all_metrics.update(m.keys())
    all_tasks = sorted(all_tasks)
    all_metrics = sorted(all_metrics, key=metric_sort_key)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'task_idx', 'task_name'] + [f'seed_{s}' for s in seeds])
        for m in all_metrics:
            for tidx in all_tasks:
                tname = task_names[tidx] if tidx < len(task_names) else f"task{tidx}"
                row = [m, tidx, tname]
                for _, task_metrics in per_seed:
                    val = task_metrics.get(tidx, {}).get(m, None)
                    row.append(f"{val:.6f}" if val is not None else '')
                w.writerow(row)


def write_summary_csv(agg, task_names, out_path, decimals):
    all_metrics = set()
    for tidx in agg:
        all_metrics.update(agg[tidx].keys())
    all_metrics = sorted(all_metrics, key=metric_sort_key)
    task_indices = sorted(agg.keys())
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'task_idx', 'task_name', 'mean', 'std', 'n_seeds'])
        for m in all_metrics:
            for tidx in task_indices:
                if m not in agg[tidx]:
                    continue
                tname = task_names[tidx] if tidx < len(task_names) else f"task{tidx}"
                mean, std, n, _ = agg[tidx][m]
                w.writerow([m, tidx, tname, f"{mean:.{decimals}f}", f"{std:.{decimals}f}", n])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--result_dir', required=True,
                        help='Parent dir containing seed subdirectories with best_model_results_*.txt')
    parser.add_argument('--task_names', default=None,
                        help='Comma-separated task names (e.g. ihm,risk). If omitted, parsed from parent dir name.')
    parser.add_argument('--decimals', type=int, default=3)
    parser.add_argument('--ddof', type=int, default=0,
                        help='Delta degrees of freedom for std (0=population, 1=sample). Default 0.')
    args = parser.parse_args()

    rd = args.result_dir
    if not os.path.isdir(rd):
        print(f"[error] result_dir does not exist: {rd}", file=sys.stderr)
        sys.exit(1)

    task_names = (args.task_names.split(',') if args.task_names
                  else task_names_from_dir(rd))

    seed_dirs = find_seed_dirs(rd)
    if not seed_dirs:
        print(f"[error] no seed subdirectories with best_model_results_*.txt under {rd}", file=sys.stderr)
        sys.exit(1)

    per_seed = []
    for seed, sdir in seed_dirs:
        files = sorted(glob.glob(os.path.join(sdir, 'best_model_results_*.txt')))
        if not files:
            continue
        latest = max(files, key=os.path.getmtime)
        parsed = parse_best_block(latest)
        if not parsed:
            print(f"[warn] no 'Final Best Model Test' block in {latest}", file=sys.stderr)
            continue
        per_seed.append((seed, parsed))

    if not per_seed:
        print("[error] no parseable results found", file=sys.stderr)
        sys.exit(1)

    print(f"Found results for {len(per_seed)} seed(s): {[s for s, _ in per_seed]}")
    print(f"Task names (in order): {task_names}")

    agg = aggregate(per_seed, ddof=args.ddof)
    n_seeds = len(per_seed)

    out_compact = os.path.join(rd, 'aggregated_table.tex')
    out_full = os.path.join(rd, 'aggregated_table_full.tex')
    out_per_seed_tex = os.path.join(rd, 'aggregated_per_seed_table.tex')
    out_per_seed = os.path.join(rd, 'aggregated_per_seed.csv')
    out_summary = os.path.join(rd, 'aggregated_summary.csv')

    with open(out_compact, 'w') as f:
        f.write(make_compact_table(agg, task_names, n_seeds, args.decimals) + '\n')
    with open(out_full, 'w') as f:
        f.write(make_full_table(agg, task_names, n_seeds, args.decimals) + '\n')
    with open(out_per_seed_tex, 'w') as f:
        f.write(make_per_seed_table(per_seed, agg, task_names, args.decimals) + '\n')
    write_per_seed_csv(per_seed, task_names, out_per_seed)
    write_summary_csv(agg, task_names, out_summary, args.decimals)

    print(f"Wrote: {out_compact}")
    print(f"Wrote: {out_full}")
    print(f"Wrote: {out_per_seed_tex}")
    print(f"Wrote: {out_per_seed}")
    print(f"Wrote: {out_summary}")


if __name__ == '__main__':
    main()
