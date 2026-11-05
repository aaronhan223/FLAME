"""Spearman statistics for App. D.6 (robustness to modality corruption).

Recomputes, from the raw sweep JSON written by
src/analysis/eval_modality_heterogeneity.py (Setup 4, first_appearance FLAME-CL,
seed 42, 186 conditions over 31 (task, modality) pairs), every correlation the
D.6 text states without numbers:

  * routing reaction vs. modality importance (the "uncorrelated" claim)
  * learned fusion share vs. modality importance (the "correlates positively" claim),
    pooled over all 31 pairs and per task with exact permutation p-values
  * JSD monotonicity in the noise level, cross-modality spillover, Tab. 17 medians

Importance of modality m on task k = AUROC lost when m is removed
(clean - zero_1.0), i.e. -ablation_drop. Routing reaction = Jensen-Shannon
divergence of m's own gate distribution from its clean gate distribution.

Outputs tables/robustness_spearman.{tex,md} and data/robustness_pairs.csv.

    ~/anaconda3/envs/mmt/bin/python make_robustness_stats.py
"""

import csv
import glob
import itertools
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON = glob.glob(
    '/cis/home/schaud35/clinical-highmmt/src/analysis/analysis_results/heterogeneity/continual/'
    'laplace/pheno-density__los-birads-mortality__ihm-risk-readmission/*/42/'
    'cl_ours_target_moe_and_encoder_*_wd1.0_mod_drop_rate_0.0/modality_heterogeneity.json')
assert len(JSON) == 1, JSON
NOISE = ['gauss_0.5', 'gauss_1.0', 'gauss_2.0', 'gauss_4.0']


def exact_perm_p(x, y):
    """Two-sided exact permutation p-value for Spearman rho (all n! orderings)."""
    rho = spearmanr(x, y).correlation
    perms = [spearmanr(x, p).correlation for p in itertools.permutations(y)]
    return rho, np.mean([abs(r) >= abs(rho) - 1e-12 for r in perms])


def main():
    d = json.load(open(JSON[0]))
    tasks = d['results']['ours']['tasks']
    rows = []
    for tname, T in tasks.items():
        for mod, M in T['modalities'].items():
            c = M['conditions']
            rows.append(dict(
                task=tname, modality=mod,
                contribution_share=M['contribution_share'], magnitude_share=M['magnitude_share'],
                importance=-M['ablation_drop'],
                **{f'jsd_{k}': c[k]['self_routing_jsd'] for k in NOISE + ['shuffle_1.0']},
                spill_max=max(c[k]['other_routing_jsd'] for k in c),
                rel_noise4=c['gauss_4.0']['rel_change'], rel_shuffle=c['shuffle_1.0']['rel_change'],
                rel_zero=c['zero_1.0']['rel_change']))
    n_pairs = len(rows)
    n_cond = n_pairs * 6
    col = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    imp = col('importance')

    # ---- pooled correlations (n = 31)
    out = []

    def add(name, x, y, claim):
        rho, p = spearmanr(x, y)
        r, pr = pearsonr(x, y)
        out.append(dict(name=name, rho=rho, p=p, r=r, pr=pr, n=len(x), claim=claim))

    add('Routing JSD at 2x noise vs. importance', col('jsd_gauss_2.0'), imp,
        'routing reaction is uncorrelated with importance')
    add('Routing JSD at 4x noise vs. importance', col('jsd_gauss_4.0'), imp, 'same, strongest noise')
    add('Routing JSD under shuffling vs. importance', col('jsd_shuffle_1.0'), imp,
        'same, cross-modal disagreement')
    add('Routing JSD at 2x noise vs. contribution share', col('jsd_gauss_2.0'),
        col('contribution_share'), 'reaction vs. learned reliance')
    add('Fusion contribution share vs. importance', col('contribution_share'), imp,
        'fusion head allocates weight by informativeness (headline)')
    add('Fusion magnitude share vs. importance', col('magnitude_share'), imp,
        'same, |contribution| instead of its spread')

    # ---- per-task share vs importance, exact p
    per_task = []
    for tname in tasks:
        sub = [r for r in rows if r['task'] == tname]
        x = [r['contribution_share'] for r in sub]
        y = [r['importance'] for r in sub]
        rho, p = exact_perm_p(x, y)
        floor = exact_perm_p(list(range(len(x))), list(range(len(x))))[1]
        per_task.append((tname, len(sub), rho, p, floor))

    # ---- other D.6 claims
    mono = {}
    for tname in tasks:
        sub = [r for r in rows if r['task'] == tname]
        m = [np.mean([r[f'jsd_{k}'] for r in sub]) for k in NOISE]
        mono[tname] = (all(b >= a - 1e-12 for a, b in zip(m, m[1:])), m)
    mono_pairs = sum(all(r[f'jsd_{b}'] >= r[f'jsd_{a}'] - 1e-12 for a, b in zip(NOISE, NOISE[1:]))
                     for r in rows)
    spill = max(r['spill_max'] for r in rows)
    dom = {}
    for tname in tasks:
        sub = [r for r in rows if r['task'] == tname]
        dom[tname] = max(sub, key=lambda r: r['contribution_share'])
    med_noise = np.median([100 * r['rel_noise4'] for r in dom.values()])
    med_shuf = np.median([100 * r['rel_shuffle'] for r in dom.values()])

    # ---- write CSV
    os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
    with open(os.path.join(ROOT, 'data', 'robustness_pairs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- Markdown
    fmt_p = lambda p: f'{p:.1e}' if p < 1e-2 else f'{p:.2f}'  # noqa: E731
    md = ['# App. D.6 Spearman statistics (generated by make_robustness_stats.py)', '',
          f'Source: `{os.path.relpath(JSON[0], "/cis/home/schaud35/clinical-highmmt")}`', '',
          f'{n_pairs} (task, modality) pairs, {n_cond} corrupted evaluations; importance = AUROC '
          'lost when the modality is removed.', '',
          '| Quantity (pooled over all pairs) | Spearman ρ | p | Pearson r | n |', '|---|---|---|---|---|']
    md += [f"| {o['name']} | {o['rho']:+.2f} | {fmt_p(o['p'])} | {o['r']:+.2f} | {o['n']} |" for o in out]
    md += ['', '### Fusion contribution share vs. importance, per task (exact permutation p)', '',
           '| Task | n modalities | ρ | exact p | smallest attainable p |', '|---|---|---|---|---|']
    md += [f'| {t} | {n} | {r:+.2f} | {p:.3f} | {fl:.3f} |' for t, n, r, p, fl in per_task]
    n_one = sum(abs(r - 1) < 1e-9 for _, _, r, _, _ in per_task)
    n_pos = sum(r > 0 for _, _, r, _, _ in per_task)
    md += ['', '### Other D.6 claims', '',
           f'- ρ = +1 on {n_one} of {len(per_task)} tasks; ρ > 0 on {n_pos} of {len(per_task)}. '
           f'Only {sum(p < 0.05 for *_, p, _ in per_task)} task(s) reach p < 0.05 alone '
           '(3 modalities cannot: the smallest attainable p is 0.333).',
           f'- Task-mean routing JSD non-decreasing in noise level (0.5→1→2→4×): '
           f'{sum(v[0] for v in mono.values())} of {len(mono)} tasks; per (task, modality): '
           f'{mono_pairs} of {n_pairs} pairs.',
           '  ' + '; '.join(f'{t}: ' + '→'.join(f'{x:.4f}' for x in v[1]) for t, v in mono.items()),
           f'- Cross-modality routing spillover (max other-modality JSD over all {n_cond} '
           f'conditions): {spill:.3g}.',
           f'- Tab. 17 medians over dominant modalities: 4× noise {med_noise:.1f}%, '
           f'shuffled {med_shuf:.1f}%.', '']
    open(os.path.join(ROOT, 'tables', 'robustness_spearman.md'), 'w').write('\n'.join(md))

    # ---- LaTeX (one compact table + per-task row)
    hd = out[4]
    rj = out[0]
    tex = [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
           r'\caption{Rank correlations behind App.~\ref{app:robustness} (Setup 4 \textsc{FLAME-CL}, '
           rf'{n_pairs} (task, modality) pairs, {n_cond} corrupted evaluations). Importance is the '
           r'AUROC lost when the modality is removed; routing reaction is the Jensen--Shannon '
           r'divergence of the gate distribution of a modality from its clean one. Per-task '
           r'$p$-values are exact permutation tests ($n=3$--$5$ modalities).}',
           r'\label{tab:robustness-spearman}',
           r'\begin{tabular}{lccc}', r'\toprule',
           r'Pooled over all pairs & Spearman $\rho$ & $p$ & Pearson $r$ \\', r'\midrule']
    for o in out:
        p = o['p']
        ps = f'{p:.2f}' if p >= 1e-2 else (r'$%.1f\times10^{%d}$' % (p / 10 ** np.floor(np.log10(p)),
                                                                   np.floor(np.log10(p))))
        name = o['name'].replace('x noise', r'$\times$ noise')
        tex.append(f"{name} & ${o['rho']:+.2f}$ & {ps} & ${o['r']:+.2f}$" + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\\[6pt]',
            r'\begin{tabular}{l' + 'c' * len(per_task) + '}', r'\toprule',
            r'Share vs.\ importance, per task & '
            + ' & '.join(t for t, *_ in per_task) + r' \\', r'\midrule',
            r'Modalities $n$ & ' + ' & '.join(str(n) for _, n, *_ in per_task) + r' \\',
            r'Spearman $\rho$ & ' + ' & '.join(f'${r:+.2f}$' for _, _, r, _, _ in per_task) + r' \\',
            r'Exact $p$ & ' + ' & '.join(f'{p:.2f}' for *_, p, _ in per_task) + r' \\',
            r'Smallest attainable $p$ & ' + ' & '.join(f'{fl:.2f}' for *_, fl in per_task) + r' \\']
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(os.path.join(ROOT, 'tables', 'robustness_spearman.tex'), 'w').write('\n'.join(tex) + '\n')

    # ---- drop-in sentences for D.6
    snippet = [
        '% Drop-in wording for App. D.6 (numbers from make_robustness_stats.py)',
        r'% "Routers do not downweight unreliable modalities; the fusion head does." paragraph:',
        (r'Routing does react to corruption, with the Jensen--Shannon divergence from the clean gate '
         r'rising with the noise level on every task, but that reaction is uncorrelated with the '
         rf"modality's importance (Spearman $\rho={rj['rho']:+.2f}$, $p={rj['p']:.2f}$, "
         rf'$n={n_pairs}$ (task, modality) pairs). Reliability weighting instead resides in the '
         r'fusion head: the learned share of each modality in the fusion logit correlates with its '
         rf"measured importance (the AUROC drop on removal) at $\rho={hd['rho']:+.2f}$ "
         rf"($p={hd['p'] / 10 ** np.floor(np.log10(hd['p'])):.1f}\times10^{{{int(np.floor(np.log10(hd['p'])))}}}$, "
         rf'$n={n_pairs}$), and per task at $\rho=+1$ on {n_one} of {len(per_task)} tasks '
         rf'(Tab.~\ref{{tab:robustness-spearman}}).'),
    ]
    open(os.path.join(ROOT, 'tables', 'robustness_d6_snippet.tex'), 'w').write('\n'.join(snippet) + '\n')
    print('\n'.join(md))
    print('\n'.join(snippet))


if __name__ == '__main__':
    main()
