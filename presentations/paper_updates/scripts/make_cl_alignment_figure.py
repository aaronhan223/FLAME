"""Continual-phase spectra and singular-vector alignment, all four CL sequences.

Rebuttal promise (reviewer 9k3u, W2): input / weight-only / data-aware spectra
and singular-vector alignment for each stage's active component, under that
stage's routed inputs, across all four sequences.

Source: src/analysis/eval_cl_stage_geometry.py, run on the reported FLAME-CL
configuration (first_appearance; Setup 2 = its only run, wd0.1) for every
stage of every sequence:
    src/analysis/analysis_results/cl_stage_geometry/ours_first_appearance/
        laplace/<sequence>/experts_5/seed<seed>_stage<s>.pt
Each file holds the stage's untruncated active component (pre-reservation
checkpoint) and the second moment C = E[z z^T] of the inputs the routers
dispatched to each expert sublayer on a fixed probe set (first <= 512
validation samples of each of the stage's tasks).

For each expert sublayer W (5 experts x {fc1, fc2}, 128 x 128) with SVD
W = U S V^T and C = Q diag(lambda) Q^T:
    input       E_x(k) = sum_{j<=k} lambda_j / sum_j lambda_j
    weight-only E_w(k) = sum_{i<=k} sigma_i^2 / sum_i sigma_i^2
    data-aware  E_f(k) = sum_{i<=k} sigma_i^2 v_i^T C v_i / sum_i (same)
    alignment   A(r)   = ||V_r^T Q_r||_F^2 / r       (random subspaces: r / 128)
the same definitions as Fig. 3 / Figs. 11-16 and the talk's S1/S2 figures.

Init (src/continual/cl_moe.py: zero_init_fc2): at stages >= 2 the new
component's fc2 is zero-initialised and fc1 / the temporal conv keep the
default random init; at stage 1 both are random (trained from scratch).
Proposition 2's zero-init premise therefore applies to fc2, so the two
sublayers are drawn separately.

Outputs
  figs/fig_cl_spectra_alignment.{pdf,png}   4 setups x {input, weight-only,
                                            data-aware, alignment}
  figs/fig_cl_alignment.{pdf,png}           alignment only, one row (for use
                                            next to Fig. 16-format spectra)
  figs/fig16_setup{n}_stage{s}.png          Fig. 16-format panels for every
                                            setup (copied from cl_spectrum_eval);
                                            s is 0-based like the PNG titles and
                                            the Fig. 16 subcaptions
  tables/tab_cl_spectra_alignment.tex, tables/cl_spectra_alignment.md
  data/cl_spectra_alignment_sublayers.csv
"""
import glob
import os
import shutil

import numpy as np
import pandas as pd
import torch

import cl_data
from cl_data import SETUPS, TASK_LABEL
import paper_style as ps
from paper_style import INK, INK_2, TEXT_W
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(cl_data.SRC, 'analysis', 'analysis_results')
GEOM = os.path.join(ANALYSIS, 'cl_stage_geometry', 'ours_first_appearance', 'laplace')
SPEC = os.path.join(ANALYSIS, 'cl_spectrum_eval', 'ours_first_appearance', 'laplace')
MAIN_SEED = 42
RANK = 32          # reserved rank r_t
D = 128            # every expert sublayer is 128 x 128
N_EXPERTS = 5

# Ordinal blue ramp (dataviz sequential blue, steps 300/500/700): stage order is
# ordinal. Validated all pairs: CVD dE >= 19.0, normal dE >= 19.2, lightest
# step 2.5:1 on white. Sublayer identity is carried by dash, not colour.
STAGE_COLOR = ['#6da7ec', '#256abf', '#0d366b']
SUB_LS = {'fc2': '-', 'fc1': (0, (3.2, 1.6))}
SUB_LW = {'fc2': 1.15, 'fc1': 0.95}
MUTED = '#8a8984'


def seq_dir(setup):
    return SETUPS[setup]['seq']


def stage_labels(setup):
    return ['+'.join(TASK_LABEL[t] for t in st) for st in SETUPS[setup]['stages']]


# ---------------------------------------------------------------- lenses --
def cumulative(p):
    p = np.clip(np.asarray(p, float), 0, None)
    return np.cumsum(p) / max(p.sum(), 1e-30)


def rank_at(p, frac):
    return int(np.searchsorted(cumulative(p), frac) + 1)


def second_moment(acc):
    return np.asarray(acc['S'], dtype=np.float64) / max(float(acc['n']), 1e-30)


def lenses(W, C):
    _, s, Vt = np.linalg.svd(W)
    lam, Q = np.linalg.eigh(C)
    lam, Q = np.clip(lam[::-1], 0, None), Q[:, ::-1]
    w = s ** 2
    f = s ** 2 * np.einsum('kd,de,ke->k', Vt, C, Vt)
    M = (Vt @ Q) ** 2                                   # M[k, j] = (v_k . q_j)^2
    A = np.diag(np.cumsum(np.cumsum(M, 0), 1)) / np.arange(1, len(s) + 1)
    return {'x': lam, 'w': w, 'f': f, 'A': A}


def load_records(setup, seed):
    recs = {}
    for p in sorted(glob.glob(os.path.join(GEOM, seq_dir(setup), 'experts_5',
                                           f'seed{seed}_stage*.pt'))):
        r = torch.load(p, map_location='cpu')
        recs[r['stage']] = r
    return recs


def kept_check(r, s_idx):
    """Kept factors (reserved ckpt) vs top-r SVD of the untruncated component."""
    worst = 0.0
    for e in range(N_EXPERTS):
        for sub in ('fc1', 'fc2'):
            sv = torch.linalg.svdvals(r['W'][f'e{e}.{sub}'].double())[:RANK]
            S = r['kept'][f'e{e}.{sub}_S'].double()
            worst = max(worst, float((sv - S).abs().max() / sv[0]))
    return worst


def collect():
    rows, curves, worst = [], {}, 0.0
    for setup in SETUPS:
        seeds = [MAIN_SEED] + (cl_data.SETUP4_TIMESTAMP_SEEDS if setup == 4 else [])
        for seed in seeds:
            recs = load_records(setup, seed)
            n_st = len(SETUPS[setup]['stages'])
            if len(recs) != n_st:
                raise FileNotFoundError(f'Setup {setup} seed {seed}: {len(recs)}/{n_st} stage files '
                                        f'under {os.path.join(GEOM, seq_dir(setup))}')
            for st, r in recs.items():
                worst = max(worst, kept_check(r, st))
                for e in range(N_EXPERTS):
                    for sub in ('fc1', 'fc2'):
                        W = r['W'][f'e{e}.{sub}'].numpy().astype(np.float64)
                        C = second_moment(r['C'][f'e{e}.{sub}'])
                        L = lenses(W, C)
                        curves[(setup, seed, st, e, sub)] = L
                        rows.append({
                            'setup': setup, 'seed': seed, 'stage': st + 1,
                            'tasks': stage_labels(setup)[st], 'expert': e + 1, 'sublayer': sub,
                            'init': 'zero' if (sub == 'fc2' and st > 0) else 'random',
                            'tokens': int(r['C'][f'e{e}.{sub}']['n']),
                            'r90_input': rank_at(L['x'], 0.9), 'r99_input': rank_at(L['x'], 0.99),
                            'r90_weight': rank_at(L['w'], 0.9), 'r99_weight': rank_at(L['w'], 0.99),
                            'r90_data': rank_at(L['f'], 0.9), 'r99_data': rank_at(L['f'], 0.99),
                            'kept32_weight': float(cumulative(L['w'])[RANK - 1]),
                            'kept32_data': float(cumulative(L['f'])[RANK - 1]),
                            'align8': float(L['A'][7]), 'align16': float(L['A'][15]),
                            'align32': float(L['A'][31]),
                        })
    return pd.DataFrame(rows), curves, worst


# ---------------------------------------------------------------- figure --
def _curve(curves, setup, st, sub, key):
    cs = []
    for e in range(N_EXPERTS):
        L = curves[(setup, MAIN_SEED, st, e, sub)]
        cs.append(L['A'] if key == 'A' else cumulative(L[key]))
    return np.median(np.asarray(cs), 0)


def _style_panel(ax, key, show_x, show_y):
    ax.set_xlim(0, D)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 32, 64, 128])
    ax.set_yticks([0, 0.5, 1])
    ax.grid(False)
    ax.axvline(RANK, color=MUTED, lw=0.6, zorder=1)
    if key == 'A':
        r = np.arange(0, D + 1)
        ax.plot(r, r / D, color=MUTED, lw=0.8, ls=(0, (1, 1.3)), zorder=1)
    else:
        ax.axhline(0.9, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
    if not show_x:
        ax.tick_params(labelbottom=False)
    if not show_y:
        ax.tick_params(labelleft=False)
    ax.tick_params(pad=1.5)


COLS = [('x', 'input spectrum'), ('w', 'weight-only'), ('f', 'data-aware'), ('A', 'alignment $A(r)$')]


def Inches(fig):
    W, H = fig.get_size_inches()

    def axes(left, top, w, h):
        return fig.add_axes([left / W, 1 - (top + h) / H, w / W, h / H])

    def text(x, top, s, **kw):
        return fig.text(x / W, 1 - top / H, s, **kw)
    return axes, text


def legend_handles(n_stages=3):
    h = [Line2D([], [], color=STAGE_COLOR[i], lw=2.0, label=f'stage {i + 1}') for i in range(n_stages)]
    h += [Line2D([], [], color=INK_2, lw=SUB_LW['fc2'], ls=SUB_LS['fc2'], label='fc2 (zero-init at stages $\\geq$2)'),
          Line2D([], [], color=INK_2, lw=SUB_LW['fc1'], ls=SUB_LS['fc1'], label='fc1 (random init)'),
          Line2D([], [], color=MUTED, lw=0.8, ls=(0, (1, 1.3)), label='random, $r/128$')]
    return h


def fig_grid(curves):
    LEFT, RIGHT = 0.34, 0.06
    GAP, GAP_A = 0.17, 0.36          # alignment column gets its own y tick labels
    PW = (TEXT_W - LEFT - RIGHT - 2 * GAP - GAP_A) / 4
    PH, HEAD, ROWGAP = 0.74, 0.17, 0.07
    TOP = 0.50                       # legend + column titles
    BOTTOM = 0.36
    H = TOP + 4 * (HEAD + PH + ROWGAP) - ROWGAP + BOTTOM
    fig = plt.figure(figsize=(TEXT_W, H))
    axes, text = Inches(fig)
    xs = [LEFT, LEFT + PW + GAP, LEFT + 2 * (PW + GAP), LEFT + 3 * PW + 2 * GAP + GAP_A]
    k = np.arange(0, D + 1)
    for i, setup in enumerate(SETUPS):
        top = TOP + i * (HEAD + PH + ROWGAP)
        labs = stage_labels(setup)
        text(LEFT, top + 0.02, f'Setup {setup}', ha='left', va='top', fontsize=7,
             fontweight='bold', color=INK)
        text(LEFT + 0.46, top + 0.02, '  $\\rightarrow$  '.join(labs), ha='left', va='top',
             fontsize=6.5, color=INK_2)
        for c, (key, title) in enumerate(COLS):
            ax = axes(xs[c], top + HEAD, PW, PH)
            _style_panel(ax, key, show_x=(i == 3), show_y=(c in (0, 3)))
            for st in range(len(labs)):
                for sub in ('fc1', 'fc2'):
                    y = _curve(curves, setup, st, sub, key)
                    y = np.concatenate([[0.0], y])
                    ax.plot(k, y, color=STAGE_COLOR[st], ls=SUB_LS[sub], lw=SUB_LW[sub],
                            zorder=3 if sub == 'fc2' else 2, solid_capstyle='round')
            if i == 0:
                text(xs[c] + PW / 2, TOP - 0.03, title, ha='center', va='bottom', fontsize=7,
                     color=INK)
                ax.text(RANK + 2, 0.03, '$r_t$', ha='left', va='bottom', fontsize=6, color=INK_2)
            if i == 3:
                ax.set_xlabel('rank $k$' if key != 'A' else 'rank $r$', labelpad=1.5)
    fig.text(0.04 / TEXT_W, 1 - (TOP + 2 * (HEAD + PH + ROWGAP) - ROWGAP / 2) / H,
             'cumulative energy (first three columns)', rotation=90, ha='left', va='center',
             fontsize=7, color=INK_2)
    fig.legend(handles=legend_handles(), loc='upper center', bbox_to_anchor=(0.5, 1 - 0.02 / H),
               ncol=6, frameon=False, handlelength=1.9, columnspacing=1.1, handletextpad=0.4,
               fontsize=6.3)
    return fig


def fig_alignment_row(curves):
    LEFT, RIGHT, GAP = 0.36, 0.06, 0.17
    PW = (TEXT_W - LEFT - RIGHT - 3 * GAP) / 4
    PH, TOP, BOTTOM = 0.95, 0.40, 0.34
    H = TOP + PH + BOTTOM
    fig = plt.figure(figsize=(TEXT_W, H))
    axes, text = Inches(fig)
    k = np.arange(0, D + 1)
    for i, setup in enumerate(SETUPS):
        x0 = LEFT + i * (PW + GAP)
        ax = axes(x0, TOP, PW, PH)
        _style_panel(ax, 'A', show_x=True, show_y=(i == 0))
        labs = stage_labels(setup)
        for st in range(len(labs)):
            for sub in ('fc1', 'fc2'):
                y = np.concatenate([[0.0], _curve(curves, setup, st, sub, 'A')])
                ax.plot(k, y, color=STAGE_COLOR[st], ls=SUB_LS[sub], lw=SUB_LW[sub],
                        zorder=3 if sub == 'fc2' else 2)
        ax.set_xlabel('rank $r$', labelpad=1.5)
        if i == 0:
            ax.set_ylabel('alignment $A(r)$', labelpad=2)
        text(x0, TOP - 0.05, f'Setup {setup}', ha='left', va='bottom', fontsize=7,
             fontweight='bold', color=INK)
        ax.text(RANK + 2, 0.03, '$r_t$', ha='left', va='bottom', fontsize=6, color=INK_2)
    fig.legend(handles=legend_handles(), loc='upper center', bbox_to_anchor=(0.5, 1 - 0.02 / H),
               ncol=6, frameon=False, handlelength=1.9, columnspacing=1.1, handletextpad=0.4,
               fontsize=6.3)
    return fig


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(ROOT, 'figs', f'{name}.{ext}'))
    plt.close(fig)
    print('wrote', os.path.join(ROOT, 'figs', name + '.{pdf,png}'))


# ----------------------------------------------------------------- table --
def _med(g, col, fmt):
    return fmt.format(np.median(g[col]))


def tables(df):
    d42 = df[df.seed == MAIN_SEED]
    tex = [r'\begin{table}[t]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{3.2pt}',
           r'\caption{Continual-phase spectra and singular-vector alignment of the active component '
           r'of each stage (FLAME-CL, seed 42), median over the 5 experts, reported as fc2\,/\,fc1. '
           r'$r_{90}$: rank at 90\% cumulative energy (ambient 128). Kept: data-aware energy inside '
           r'the top-32 singular directions, i.e.\ what survives the $r_t{=}32$ truncation. '
           r'$A(r)=\lVert V_r^\top Q_r\rVert_F^2/r$ between the top-$r$ right singular vectors of '
           r'the component and the top-$r$ eigenvectors of its routed-input second moment; random '
           r'subspaces give $r/128$ (0.125 at $r{=}16$, 0.25 at $r{=}32$). fc2 is zero-initialised at '
           r'stages $\geq$2; fc1, and both sublayers at stage 1, start from random init.}',
           r'\label{tab:cl-spectra-alignment}',
           r'\begin{tabular}{@{}llccccc@{}}', r'\toprule',
           r' & & \multicolumn{3}{c}{$r_{90}$ (fc2\,/\,fc1)} & Kept at $r_t{=}32$ & $A(16)$ \\',
           r'\cmidrule(lr){3-5}',
           r'Setup & Stage & input & weight-only & data-aware & data-aware & (random 0.125) \\',
           r'\midrule']
    md = ['# Continual-phase spectra and singular-vector alignment (FLAME-CL, first_appearance)', '',
          'Median over the 5 experts; each cell is **fc2 / fc1**. Seed 42 for every setup; Setup 4 '
          'also over 4 seeds below. r90 = rank at 90% cumulative energy (ambient 128). '
          'Kept = data-aware energy in the top-32 singular directions. A(r) = ||V_r^T Q_r||_F^2 / r '
          '(random subspaces: r/128).', '',
          '| Setup | Stage | tasks | r90 input | r90 weight | r90 data-aware | kept@32 data-aware | '
          'A(8) | A(16) | A(32) |', '|---|---|---|---|---|---|---|---|---|---|']
    for setup in SETUPS:
        g_s = d42[d42.setup == setup]
        n_st = g_s.stage.max()
        for st in range(1, n_st + 1):
            g = g_s[g_s.stage == st]
            a, b = g[g.sublayer == 'fc2'], g[g.sublayer == 'fc1']

            def pair(col, fmt):
                return f'{_med(a, col, fmt)}\\,/\\,{_med(b, col, fmt)}'

            def mpair(col, fmt):
                return f'{_med(a, col, fmt)} / {_med(b, col, fmt)}'
            tasks = g.tasks.iloc[0]
            first = f'{setup}' if st == 1 else ''
            tex.append(f'{first} & {st}: {tasks} & {pair("r90_input", "{:.0f}")} & '
                       f'{pair("r90_weight", "{:.0f}")} & {pair("r90_data", "{:.0f}")} & '
                       f'{pair("kept32_data", "{:.2f}")} & {pair("align16", "{:.2f}")} \\\\')
            md.append(f'| {setup} | {st} | {tasks} | {mpair("r90_input", "{:.0f}")} | '
                      f'{mpair("r90_weight", "{:.0f}")} | {mpair("r90_data", "{:.0f}")} | '
                      f'{mpair("kept32_data", "{:.3f}")} | {mpair("align8", "{:.2f}")} | '
                      f'{mpair("align16", "{:.2f}")} | {mpair("align32", "{:.2f}")} |')
        if setup != 4:
            tex.append(r'\addlinespace[2pt]')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    d4 = df[df.setup == 4]
    md += ['', '## Setup 4 across seeds (42, 703, 8622, 92784)', '',
           'Median over 5 experts x 4 seeds, [min, max] over the 4 per-seed medians.', '',
           '| Stage | sublayer | r90 data-aware | kept@32 data-aware | A(16) |', '|---|---|---|---|---|']
    for st in range(1, 4):
        for sub in ('fc2', 'fc1'):
            g = d4[(d4.stage == st) & (d4.sublayer == sub)]
            per = g.groupby('seed')[['r90_data', 'kept32_data', 'align16']].median()
            md.append(f'| {st} | {sub} | {g.r90_data.median():.0f} [{per.r90_data.min():.0f}, '
                      f'{per.r90_data.max():.0f}] | {g.kept32_data.median():.3f} '
                      f'[{per.kept32_data.min():.3f}, {per.kept32_data.max():.3f}] | '
                      f'{g.align16.median():.2f} [{per.align16.min():.2f}, {per.align16.max():.2f}] |')

    zero = d42[d42.init == 'zero']
    rnd_later = d42[(d42.sublayer == 'fc1') & (d42.stage > 1)]
    first = d42[d42.stage == 1]
    summary = {
        'zero-init fc2 (stages>=2): r90 data-aware': f'{zero.r90_data.median():.0f} '
                                                      f'[{zero.r90_data.min()}, {zero.r90_data.max()}]',
        'zero-init fc2 (stages>=2): kept@32 data-aware': f'{zero.kept32_data.median():.3f} '
                                                         f'[min {zero.kept32_data.min():.3f}]',
        'zero-init fc2 (stages>=2): A(16)': f'{zero.align16.median():.2f} '
                                            f'[{zero.align16.min():.2f}, {zero.align16.max():.2f}]',
        'random-init fc1 (stages>=2): r90 data-aware': f'{rnd_later.r90_data.median():.0f} '
                                                        f'[{rnd_later.r90_data.min()}, {rnd_later.r90_data.max()}]',
        'random-init fc1 (stages>=2): kept@32 data-aware': f'{rnd_later.kept32_data.median():.3f} '
                                                           f'[min {rnd_later.kept32_data.min():.3f}]',
        'random-init fc1 (stages>=2): A(16)': f'{rnd_later.align16.median():.2f} '
                                              f'[{rnd_later.align16.min():.2f}, {rnd_later.align16.max():.2f}]',
        'stage 1, both sublayers: r90 data-aware': f'{first.r90_data.median():.0f} '
                                                   f'[{first.r90_data.min()}, {first.r90_data.max()}]',
        'stage 1, both sublayers: kept@32 data-aware': f'{first.kept32_data.median():.3f}',
        'stage 1, both sublayers: A(16)': f'{first.align16.median():.2f}',
        'all sublayers: r90 input': f'{d42.r90_input.median():.0f} [{d42.r90_input.min()}, {d42.r90_input.max()}]',
    }
    md += ['', '## Pooled over all setups (seed 42, per sublayer: median [range])', '']
    md += [f'- {k}: {v}' for k, v in summary.items()]
    with open(os.path.join(ROOT, 'tables', 'tab_cl_spectra_alignment.tex'), 'w') as f:
        f.write('\n'.join(tex) + '\n')
    with open(os.path.join(ROOT, 'tables', 'cl_spectra_alignment.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))


# ------------------------------------------------- Fig. 16-format panels --
def copy_fig16_panels():
    """Fig. 16 = eval_cl_spectrum's per-stage PNGs (seed 42, 10 test batches).
    Copy them for every setup so Fig. 16 can be extended in its own format."""
    n = 0
    for setup in SETUPS:
        for st, stage in enumerate(SETUPS[setup]['stages']):
            src = os.path.join(SPEC, seq_dir(setup), 'experts_5', str(MAIN_SEED),
                               f'stage{st}_{"-".join(stage)}', 'expert_input_spectrum_comparison.png')
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(ROOT, 'figs', f'fig16_setup{setup}_stage{st}.png'))
                n += 1
            else:
                print('missing Fig. 16-format panel:', src)
    print(f'copied {n} Fig. 16-format panels')


def main():
    ps.apply()
    os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
    df, curves, worst = collect()
    print(f'kept factors vs top-{RANK} SVD of the untruncated component: '
          f'max rel. singular-value diff {worst:.1e}')
    df.to_csv(os.path.join(ROOT, 'data', 'cl_spectra_alignment_sublayers.csv'), index=False)
    save(fig_grid(curves), 'fig_cl_spectra_alignment')
    save(fig_alignment_row(curves), 'fig_cl_alignment')
    tables(df)
    copy_fig16_panels()


if __name__ == '__main__':
    main()
