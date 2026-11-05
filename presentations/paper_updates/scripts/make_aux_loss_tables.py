"""Per-task auxiliary-loss weight tables (App. D.2.5; promised in the rebuttal, Q2).

Reads the raw per-seed result files of run_flame_aux_ablation*.sh through the
loader in src/make_aux_ablation_tables.py (same parser, same task-index order,
which places readmission before mortality in the result files), then writes

  tables/tab_aux_loss_per_task.tex   two tables: lambda_bal sweep, lambda_int sweep
  tables/aux_loss_per_task.md        Markdown copy + the range check quoted in D.2.5
  data/aux_loss_per_seed.csv         every (config, seed, task) AUROC

Weights are reported in the paper's convention. The code's --balance_loss_coef
multiplies a fixed 1e-2 inside the MoE (fusemoe_multitask/moe.py: balance_loss *
loss_coef(=1e-2) * balance_coef), so lambda_bal = 0.01 * balance_coef: the code
sweep {0, 0.1, 1, 10, 100} is lambda_bal in {0, 0.001, 0.01, 0.1, 1} (Tab. 12).
lambda_int is the diversity coefficient as is. Mean +/- sample std (ddof=1) over
seeds 42, 20492, 28411; the 9-task mean is averaged within a seed first.

    ~/anaconda3/envs/mmt/bin/python make_aux_loss_tables.py
"""

import csv
import os
import sys

SRC = '/cis/home/schaud35/clinical-highmmt/src'
sys.path.insert(0, SRC)
from make_aux_ablation_tables import (  # noqa: E402
    find_configs, scores_for_config, resolve_task_order, cell_stats, overall_stats,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK_SPEC = 'ihm-los-pheno-mortality-readmission-birads-risk-density-diag'
ROW_ORDER = ['ihm', 'los', 'pheno', 'mortality', 'readmission', 'birads', 'risk', 'density', 'diag']
LABEL = {'ihm': '48-IHM', 'los': 'LOS', 'pheno': '25-PHENO', 'mortality': 'MOR',
         'readmission': 'RAD', 'birads': 'BIRADS', 'risk': 'RISK', 'density': 'DENSITY',
         'diag': 'DIAG'}
DATASET = {'ihm': 'MIMIC-IV', 'los': 'MIMIC-IV', 'pheno': 'MIMIC-IV', 'mortality': 'eICU',
           'readmission': 'eICU', 'birads': 'EMBED', 'risk': 'EMBED', 'density': 'EMBED',
           'diag': 'ADNI'}
BAL_SCALE = 0.01
DEFAULT = (1.0, 0.1)   # code coefficients of the shipped operating point
SEEDS_EXPECTED = 3


def lam(v):
    return f'{v:g}'


def main():
    task_names = resolve_task_order(TASK_SPEC)
    configs = find_configs(os.path.join(SRC, 'results'))
    scores = {k: scores_for_config(d, task_names) for k, d in configs.items()}
    for k, sc in scores.items():
        n = {len(v) for v in sc.values()}
        if n != {SEEDS_EXPECTED}:
            raise RuntimeError(f'config {k}: seeds per task {n}, expected {SEEDS_EXPECTED}')

    bal_rows = sorted(k for k in scores if k[1] == DEFAULT[1])
    int_rows = sorted(k for k in scores if k[0] == DEFAULT[0])

    # ---- per-seed CSV
    os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
    with open(os.path.join(ROOT, 'data', 'aux_loss_per_seed.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lambda_bal', 'lambda_int', 'code_balance_coef', 'seed', 'task', 'auroc'])
        for (b, d), sc in sorted(scores.items()):
            for t in ROW_ORDER:
                for seed, v in sorted(sc[t].items()):
                    w.writerow([lam(b * BAL_SCALE), lam(d), lam(b), seed, LABEL[t], f'{v:.6f}'])

    # ---- range check (D.2.5: "no task varies by more than 0.014 ... except DIAG (0.037)")
    rng = {}
    for t in ROW_ORDER:
        means = [cell_stats(sc, t)[0] for sc in scores.values()]
        rng[t] = (max(means) - min(means))
    o = [overall_stats(sc, ROW_ORDER)[0] for sc in scores.values()]

    def table(rows, which):
        head = [lam(b * BAL_SCALE) if which == 'bal' else lam(d) for b, d in rows]
        head = [h + (r'$^\dagger$' if r == DEFAULT else '') for h, r in zip(head, rows)]
        sym = r'\lambda_{\mathrm{bal}}' if which == 'bal' else r'\lambda_{\mathrm{int}}'
        fixed = (r'$\lambda_{\mathrm{int}}=0.1$' if which == 'bal'
                 else r'$\lambda_{\mathrm{bal}}=0.01$')
        cap = (rf'Per-task AUROC under the {"router-balancing" if which == "bal" else "interaction"}'
               rf' loss weight sweep (joint nine-task training, {fixed}; mean $\pm$ std over seeds '
               r'42, 20492, 28411). $^\dagger$ default. Bold: best weight per task. '
               r'Range: max $-$ min of the task mean across this sweep.')
        L = [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
             rf'\caption{{{cap}}}', rf'\label{{tab:aux-{which}-per-task}}',
             r'\begin{tabular}{ll' + 'c' * len(rows) + 'c}', r'\toprule',
             rf' & & \multicolumn{{{len(rows)}}}{{c}}{{${sym}$}} & \\',
             rf'\cmidrule(lr){{3-{len(rows) + 2}}}',
             'Dataset & Task & ' + ' & '.join(head) + r' & Range \\', r'\midrule']
        M = [f'| Task | ' + ' | '.join(h.replace(r'$^\dagger$', '†') for h in head) + ' | Range |',
             '|---' * (len(rows) + 2) + '|']
        prev = None
        for t in ROW_ORDER:
            st = [cell_stats(scores[r], t) for r in rows]
            best = max(m for m, _, _ in st)
            cells, mcells = [], []
            for m, sd, _ in st:
                c = f'{m:.3f}{{\\scriptsize$\\pm${sd:.3f}}}'
                cells.append(r'\textbf{' + c + '}' if abs(m - best) < 1e-12 else c)
                mc = f'{m:.3f}±{sd:.3f}'
                mcells.append(f'**{mc}**' if abs(m - best) < 1e-12 else mc)
            r_ = max(m for m, _, _ in st) - min(m for m, _, _ in st)
            ds = DATASET[t] if DATASET[t] != prev else ''
            if prev is not None and ds:
                L.append(r'\addlinespace[2pt]')
            prev = DATASET[t]
            L.append(f'{ds} & {LABEL[t]} & ' + ' & '.join(cells) + f' & {r_:.3f}' + r' \\')
            M.append(f'| {LABEL[t]} | ' + ' | '.join(mcells) + f' | {r_:.3f} |')
        ov = [overall_stats(scores[r], ROW_ORDER) for r in rows]
        best = max(m for m, _, _ in ov)
        cells = [(r'\textbf{%s}' if abs(m - best) < 1e-12 else '%s')
                 % f'{m:.3f}{{\\scriptsize$\\pm${sd:.3f}}}' for m, sd, _ in ov]
        r_ = max(m for m, _, _ in ov) - min(m for m, _, _ in ov)
        L += [r'\midrule', r'\multicolumn{2}{l}{Mean (9 tasks)} & ' + ' & '.join(cells)
              + f' & {r_:.3f}' + r' \\', r'\bottomrule', r'\end{tabular}', r'\end{table}']
        M.append('| **Mean (9 tasks)** | ' + ' | '.join(f'{m:.3f}±{sd:.3f}' for m, sd, _ in ov)
                 + f' | {r_:.3f} |')
        return '\n'.join(L), '\n'.join(M)

    tb, mb = table(bal_rows, 'bal')
    ti, mi = table(int_rows, 'int')
    os.makedirs(os.path.join(ROOT, 'tables'), exist_ok=True)
    open(os.path.join(ROOT, 'tables', 'tab_aux_loss_per_task.tex'), 'w').write(tb + '\n\n' + ti + '\n')
    worst = max((t for t in ROW_ORDER if t != 'diag'), key=lambda t: rng[t])
    md = ['# Auxiliary-loss weight sweep, per task (generated by make_aux_loss_tables.py)', '',
          f'λ_bal = 0.01 × code `--balance_loss_coef`; λ_int = code diversity coef. '
          f'Mean ± sample std over 3 seeds.', '',
          '## λ_bal sweep (λ_int = 0.1)', '', mb, '', '## λ_int sweep (λ_bal = 0.01)', '', mi, '',
          '## Range check across all 8 configurations (both sweeps)', '',
          '| Task | max − min of mean AUROC |', '|---|---|']
    md += [f'| {LABEL[t]} | {rng[t]:.4f} |' for t in ROW_ORDER]
    md += ['', f'Largest non-DIAG range: {LABEL[worst]} {rng[worst]:.4f}; DIAG {rng["diag"]:.4f}; '
           f'9-task mean spans {min(o):.4f}–{max(o):.4f}.', '']
    open(os.path.join(ROOT, 'tables', 'aux_loss_per_task.md'), 'w').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
