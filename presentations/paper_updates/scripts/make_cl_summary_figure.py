"""Figure version of main-text Tabs. 3 and 4 (page 9), to free table space.

  (a) Setup 4, 3 timestamp seeds: AA (dot +/- std), forgetting F (bar +/- std)
      and parameters after three stages, for FLAME-CL, the four baselines and
      the three component ablations of Tab. 3.
  (b) CL-phase rank sweep of Tab. 4 / Tab. 16 (Setup 4, seed 42): AUROC when the
      active component of the task's own stage is truncated to rank r; shade =
      drop relative to full rank.

Sources (all already on disk, nothing re-run):
  methods      -> results files via cl_data.cl_summary (reproduces Tab. 14)
  ablations    -> src/analysis/analysis_results/cl_cursor_ablation/
                  setup4_first_appearance (rank 32) and setup4_fullrank (no truncation)
  rank sweep   -> src/analysis/analysis_results/cl_lowrank_eval/ours_first_appearance

Writes figs/fig_cl_summary_tab3_tab4.{pdf,png} and data/cl_summary_tab3_tab4.csv.
"""

import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cl_data as D  # noqa: E402
import paper_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AR = '/cis/home/schaud35/clinical-highmmt/src/analysis/analysis_results'
SEEDS = D.SETUP4_TIMESTAMP_SEEDS
ORDER = ['pheno', 'density', 'los', 'birads', 'mortality', 'ihm', 'risk', 'readmission']
PRIOR = ['pheno', 'density', 'los', 'birads', 'mortality']


def _total_params_m(method, seed=42):
    p = D.parse_progression(os.path.join(D.run_dir(4, method, seed), 'param_count_progression.txt'))
    return p[max(p)]['total'] / 1e6


def methods_rows():
    rows = []
    for m in ['ours', 'trge', 'same', 'lwf', 'ewc']:
        sm = [D.cl_summary(4, m, s) for s in SEEDS]
        rows.append(dict(key=m, label=D.DISPLAY[m], group='method',
                         AA=[x['AA'] for x in sm], F=[x['F'] for x in sm],
                         params=_total_params_m(m)))
    return rows


def _own_stage_ref(run_key, seed):
    """Deployed (post-truncation) value of each prior task at its own stage."""
    tr = D.trajectory(4, run_key, seed, which='reserved')
    fs = D.first_stage(4)
    return {t: tr[t][fs[t] - 1] for t in PRIOR}


def ablation_rows():
    out = []
    spec = [('setup4_first_appearance', 'normal', 'ours', None),
            ('setup4_first_appearance', 'ablated', 'ours', '− cursor'),
            ('setup4_fullrank', 'normal', 'ours_fullrank', '− low-rank'),
            ('setup4_fullrank', 'ablated', 'ours_fullrank', '− low-rank − cursor')]
    for sub, setting, run_key, label in spec:
        AA, F, params = [], [], None
        for seed in SEEDS:
            j = json.load(open(os.path.join(AR, 'cl_cursor_ablation', sub,
                                            f'cursor_ablation_seed{seed}.json')))
            fin = {t: j['tasks'][t]['settings'][setting]['primary_metric'] for t in ORDER}
            ref = _own_stage_ref(run_key, seed)
            AA.append(np.mean([fin[t] for t in ORDER]))
            F.append(np.mean([ref[t] - fin[t] for t in PRIOR]))
            params = j['param_counts']['total'] / 1e6
        if label is None:   # sanity: the "normal" rank-32 setting must equal the method row
            sm = [D.cl_summary(4, 'ours', s)['AA'] for s in SEEDS]
            assert max(abs(a - b) for a, b in zip(AA, sm)) < 2e-3, (AA, sm)
            continue
        out.append(dict(key='abl', label=label, group='ablation', AA=AA, F=F, params=params))
    return out


def rank_sweep():
    f = glob.glob(os.path.join(AR, 'cl_lowrank_eval/ours_first_appearance/laplace/*/experts_5/42/'
                               'cl_lowrank_results.json'))
    assert len(f) == 1, f
    L = json.load(open(f[0]))
    ranks = ['0', '2', '4', '8', '16', '32', '64', 'full']
    vals = {}
    for st in L['stages'].values():
        for r in ranks:
            for tkey, v in st['ranks'][r]['tasks'].items():
                vals.setdefault(tkey, {})[r] = v['primary_metric']
    key = {D.TASK_LABEL[t]: t for t in ORDER}
    rows = [key[k] for k in ['PHENO', 'DENSITY', 'LOS', 'BIRADS', 'MOR', 'IHM', 'RISK', 'RAD']]
    lab2json = {'PHENO': 'PHENO', 'DENSITY': 'DENSITY', 'LOS': 'LOS', 'BIRADS': 'BIRADS',
                'MOR': 'MOR', 'IHM': 'IHM', 'RISK': 'RISK', 'RAD': 'RAD'}
    M = np.array([[vals[lab2json[D.TASK_LABEL[t]]][r] for r in ranks] for t in rows])
    return rows, ranks, M


def main():
    S.apply()
    rows = methods_rows() + ablation_rows()
    t_rows, ranks, M = rank_sweep()

    W, H = S.TEXT_W, 2.25
    fig = plt.figure(figsize=(W, H))
    ax_ = lambda l, t, w, h: fig.add_axes([l / W, 1 - (t + h) / H, w / W, h / H])  # noqa: E731
    top, ph = 0.36, H - 0.36 - 0.30
    n = len(rows)
    ypos = []
    y = 0
    for i, r in enumerate(rows):
        if i and r['group'] != rows[i - 1]['group']:
            y += 0.6
        ypos.append(y)
        y += 1
    ymax = y - 1

    def colour(r):
        return S.METHOD_COLOR['ours'] if r['key'] in ('ours', 'abl') else S.METHOD_COLOR[r['key']]

    # ---- (a1) AA dot plot
    a1 = ax_(0.94, top, 0.90, ph)
    for r, yy in zip(rows, ypos):
        m, sd = np.mean(r['AA']), np.std(r['AA'], ddof=1)
        hollow = r['key'] == 'abl'
        a1.errorbar(m, yy, xerr=sd, fmt='o', ms=4.2 if r['key'] == 'ours' else 3.6,
                    color=colour(r), mfc='white' if hollow else colour(r), mec=colour(r),
                    mew=0.8, elinewidth=0.7, capsize=1.5, zorder=3)
        a1.text(m + sd + 0.003, yy, f'{m:.3f}', va='center', ha='left', fontsize=5.6, color=S.INK_2)
    a1.set_xlim(0.722, 0.822)
    a1.set_xticks([0.74, 0.77, 0.80])
    a1.set_yticks(ypos)
    a1.set_yticklabels([r['label'] for r in rows])
    for lbl, r in zip(a1.get_yticklabels(), rows):
        if r['key'] == 'ours':
            lbl.set_fontweight('bold')
            lbl.set_color(S.INK)
    a1.set_ylim(ymax + 0.6, -0.6)
    a1.grid(axis='x', color=S.GRID, lw=0.5)
    a1.grid(axis='y', visible=False)
    a1.axvline(np.mean(rows[0]['AA']), color=S.METHOD_COLOR['ours'], lw=0.5, ls=(0, (2, 2)),
               zorder=1)
    a1.set_title('AA ↑', fontsize=6.6, color=S.INK, pad=3)
    abl_y = ypos[[r['group'] for r in rows].index('ablation')]
    a1.text(-1.02, abl_y - 0.62, 'FLAME-CL ablations', transform=a1.get_yaxis_transform(),
            fontsize=5.6, color=S.INK_2, style='italic', ha='left', va='center')

    # ---- (a2) forgetting bars
    a2 = ax_(1.92, top, 0.60, ph)
    for r, yy in zip(rows, ypos):
        m, sd = np.mean(r['F']), np.std(r['F'], ddof=1)
        hollow = r['key'] == 'abl'
        a2.barh(yy, m, height=0.62, color='white' if hollow else colour(r), edgecolor=colour(r),
                linewidth=0.8, hatch='////' if hollow else None, zorder=2)
        if sd > 0:
            a2.errorbar(m, yy, xerr=sd, fmt='none', ecolor=S.INK_2, elinewidth=0.6, capsize=1.3,
                        zorder=3)
        a2.text(max(m + sd, 0) + 0.003, yy, f'{max(m, 0):.3f}', va='center', ha='left',
                fontsize=5.6, color=S.INK_2)
    a2.set_xlim(0, 0.125)
    a2.set_xticks([0, 0.05, 0.10])
    a2.set_xticklabels(['0', '.05', '.10'])
    a2.set_yticks(ypos)
    a2.set_yticklabels([])
    a2.set_ylim(ymax + 0.6, -0.6)
    a2.grid(axis='x', color=S.GRID, lw=0.5)
    a2.grid(axis='y', visible=False)
    a2.set_title('F ↓', fontsize=6.6, color=S.INK, pad=3)

    # ---- (a3) parameter column
    a3 = ax_(2.54, top, 0.40, ph)
    a3.set_ylim(ymax + 0.6, -0.6)
    a3.set_xlim(0, 1)
    a3.axis('off')
    for r, yy in zip(rows, ypos):
        a3.text(0.5, yy, f"{r['params']:.1f}M", ha='center', va='center', fontsize=5.9,
                color=S.INK if r['key'] == 'ours' else S.INK_2,
                fontweight='bold' if r['key'] == 'ours' else 'normal')
    a3.set_title('Params', fontsize=6.6, color=S.INK, pad=3)

    # ---- (b) rank-sweep heatmap
    drop = M[:, -1:] - M           # drop vs full rank (positive = worse)
    cmap = LinearSegmentedColormap.from_list('drop', ['#ffffff', '#cfe0f5', '#5b8fd1', '#1f4f8f'])
    b = ax_(3.62, top, W - 3.62 - 0.06, ph)
    b.imshow(np.clip(drop, 0, 0.10), cmap=cmap, vmin=0, vmax=0.10, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            d = drop[i, j]
            b.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=5.3,
                   color='white' if d > 0.06 else S.INK)
    j32 = ranks.index('32')
    b.add_patch(Rectangle((j32 - 0.5, -0.5), 1, M.shape[0], fill=False, ec=S.METHOD_COLOR['ours'],
                          lw=1.0, zorder=4))
    for yb in (1.5, 4.5):
        b.axhline(yb, color='white', lw=1.6)
    b.set_xticks(range(len(ranks)))
    b.set_xticklabels(ranks)
    b.set_yticks(range(M.shape[0]))
    fs = D.first_stage(4)
    b.set_yticklabels([f'{D.TASK_LABEL[t]} (S{fs[t]})' for t in t_rows])
    b.tick_params(length=0)
    b.tick_params(axis='y', labelsize=5.8, pad=1.5)
    b.tick_params(axis='x', labelsize=6.0, pad=1.5)
    b.grid(False)
    for sp in b.spines.values():
        sp.set_visible(False)
    b.set_xlabel('reserved rank $r$ of the task’s own stage', labelpad=2)
    b.set_title('Rank sweep: AUROC (shade = drop vs. full rank)', fontsize=6.6, color=S.INK, pad=3)

    fig.text(0.004, 1 - 0.06 / H, '(a) Setup 4, 3 seeds', fontsize=7, fontweight='bold',
             va='top', color=S.INK)
    fig.text((3.62 - 0.62) / W, 1 - 0.06 / H, '(b) Setup 4, seed 42', fontsize=7,
             fontweight='bold', va='top', color=S.INK)

    os.makedirs(os.path.join(ROOT, 'figs'), exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(ROOT, 'figs', f'fig_cl_summary_tab3_tab4.{ext}'))
    plt.close(fig)

    with open(os.path.join(ROOT, 'data', 'cl_summary_tab3_tab4.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row', 'AA_mean', 'AA_std', 'F_mean', 'F_std', 'params_M'])
        for r in rows:
            w.writerow([r['label'], f"{np.mean(r['AA']):.4f}", f"{np.std(r['AA'], ddof=1):.4f}",
                        f"{np.mean(r['F']):.4f}", f"{np.std(r['F'], ddof=1):.4f}",
                        f"{r['params']:.1f}"])
        w.writerow([])
        w.writerow(['rank'] + [D.TASK_LABEL[t] for t in t_rows])
        for j, r in enumerate(ranks):
            w.writerow([r] + [f'{M[i, j]:.4f}' for i in range(M.shape[0])])
    for r in rows:
        print(f"{r['label']:24s} AA {np.mean(r['AA']):.3f}±{np.std(r['AA'], ddof=1):.3f}  "
              f"F {np.mean(r['F']):.3f}±{np.std(r['F'], ddof=1):.3f}  {r['params']:.1f}M")


if __name__ == '__main__':
    main()
