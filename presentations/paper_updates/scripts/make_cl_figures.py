"""Camera-ready continual-learning figures.

  fig5_cl_trajectories.pdf      Fig. 5 redrawn: Simple FT / EWC / LoRA / FLAME-CL
                                (the four methods the intro, Sec. 3.2, RQ3/RQ4
                                and App. D.3.1 name). Two rows of setups; each
                                setup = AUROC panel + encoder / MoE param panels.
  fig10_cl_auprc.pdf            Fig. 10 redrawn with the same methods/encoding.
  fig_cl_new_baselines.pdf      FLAME-CL vs TRGE / SAME / LwF, Setups 1-4
                                (Setups 1-3 seed 42; Setup 4 = mean +/- std over
                                the three timestamp seeds of Tabs. 3, 13, 14).
  fig_cl_new_baselines_setup{1,2,3,4}.pdf   the same, one setup per file.

Every *_inference.pdf twin plots inference-time parameter totals instead of
method-attributable counts (see cl_data.params).

    ~/anaconda3/envs/mmt/bin/python make_cl_figures.py
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cl_data as D  # noqa: E402
import paper_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figs')

FIG5_METHODS = ['simple_ft', 'ewc', 'lora', 'ours']
NEW_METHODS = ['trge', 'same', 'lwf', 'ours']

SEQ_TITLE = {
    1: 'PHENO → LOS → IHM',
    2: 'MOR → RAD',
    3: 'DENSITY → BIRADS → RISK',
    4: 'PHENO+DENSITY → LOS+BIRADS+MOR → IHM+RISK+RAD',
}
PARAM_LABEL_SHORT = {
    'attributable': ('Enc., method-attr. (M)', 'MoE, method-attr. (M)'),
    'inference': ('Encoder (M)', 'MoE (M)'),
}
PARAM_LABEL = {
    'attributable': ('Encoder, method-attr. (M)', 'MoE, method-attr. (M)'),
    'inference': ('Encoder, at inference (M)', 'MoE, at inference (M)'),
}


LABEL_OVERRIDE = {}   # per-figure legend names (e.g. 'LoRA (MoE only)' next to its encoder variant)


def _label(m):
    return LABEL_OVERRIDE.get(m, D.DISPLAY[m])


def _offsets(methods, width=0.24):
    n = len(methods)
    return {m: (i - (n - 1) / 2) * (width / max(n - 1, 1)) for i, m in enumerate(methods)}


def _style(m, kind):
    """Line kwargs: FLAME-CL solid/thick/filled; baselines dashed/thin/hollow."""
    is_ours = m == 'ours'
    c = S.METHOD_COLOR[m]
    kw = dict(color=c, linestyle=S.METHOD_LS[m], linewidth=1.5 if is_ours else 0.9,
              zorder=5 if is_ours else 3, solid_capstyle='round', dash_capstyle='round')
    if kind == 'perf':
        kw.update(markersize=4.0 if is_ours else 3.4, markeredgewidth=0.7,
                  markerfacecolor=c if is_ours else 'white', markeredgecolor=c)
    else:
        kw.update(marker='o', markersize=3.4 if is_ours else 3.0, markeredgewidth=0.7,
                  markerfacecolor=c if is_ours else 'white', markeredgecolor=c)
    return kw


def _mean_traj(setup, m, seeds, metric):
    runs = [D.trajectory(setup, m, s, metric) for s in seeds]
    runs = [r for r in runs if r is not None]
    if not runs:
        return None, None
    mean, std = {}, {}
    for t in D.tasks(setup):
        arr = np.array([[np.nan if v is None else v for v in r[t]] for r in runs], float)
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN before first stage
            mean[t] = np.nanmean(arr, 0)
            std[t] = np.nanstd(arr, 0, ddof=1) if len(runs) > 1 else None
    return mean, std


def _direct_labels(ax, setup, finals, x, ylim, outside=False):
    """Task names at the right edge, nudged apart so they never collide."""
    items = sorted(((v, t) for t, v in finals.items() if v is not None), key=lambda z: z[0])
    if not items:
        return
    gap = 0.075 * (ylim[1] - ylim[0])
    ys = [v for v, _ in items]
    for i in range(1, len(ys)):           # push up
        ys[i] = max(ys[i], ys[i - 1] + gap)
    over = ys[-1] - (ylim[1] - 0.02 * (ylim[1] - ylim[0]))
    if over > 0:                           # shift back down if we ran off the top
        ys = [y - over for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
    tc = blended_transform_factory(ax.transAxes, ax.transData) if outside else 'data'
    for (v, t), y in zip(items, ys):
        ax.annotate(D.TASK_LABEL[t], xy=(x - 0.05, v),
                    xytext=((1.03, y) if outside else (x + 0.12, y)), textcoords=tc,
                    fontsize=5.6, color=S.INK_2, va='center', ha='left', annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color='#b9b8b2', lw=0.4,
                                    shrinkA=0, shrinkB=1.5))


def draw_perf(ax, setup, methods, seeds, metric='auroc', label_tasks=True, labels_outside=False):
    K = len(D.SETUPS[setup]['stages'])
    fs = D.first_stage(setup)
    off = _offsets(methods)
    vals, finals = [], {}
    for m in methods:
        mean, std = _mean_traj(setup, m, seeds, metric)
        if mean is None:
            print(f'  [warn] setup {setup}: no {metric} results for {m}')
            continue
        for t in D.tasks(setup):
            xs = np.arange(fs[t], K + 1) + off[m]
            ys = mean[t][fs[t] - 1:]
            kw = _style(m, 'perf')
            ax.plot(xs, ys, marker=S.TASK_MARKER[t], **kw)
            if std is not None and std[t] is not None:
                ax.errorbar(xs, ys, yerr=std[t][fs[t] - 1:], fmt='none', ecolor=kw['color'],
                            elinewidth=0.5, capsize=1.0, capthick=0.5, alpha=0.55,
                            zorder=kw['zorder'] - 1)
            vals.extend(ys.tolist())
            if m == 'ours':
                finals[t] = ys[-1]
    lo, hi = min(vals), max(vals)
    pad = 0.07 * max(hi - lo, 0.03)
    ylim = (lo - pad, hi + pad)
    ax.set_ylim(*ylim)
    ax.set_xlim(0.6, K + (0.95 if label_tasks and not labels_outside else 0.4))
    ax.set_xticks(range(1, K + 1))
    ax.set_xticklabels([f'S{k}' for k in range(1, K + 1)])
    ax.set_ylabel('AUROC' if metric == 'auroc' else 'AUPRC', labelpad=2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    if label_tasks:
        _direct_labels(ax, setup, finals, K + off['ours'], ylim, outside=labels_outside)


def draw_params(ax, setup, methods, which, kind):
    K = len(D.SETUPS[setup]['stages'])
    off = _offsets(methods)
    top = 0
    for m in methods:
        p = D.params(setup, m, 42, kind)
        if p is None:
            continue
        ys = [p[k][which] for k in range(K)]
        ax.plot(np.arange(1, K + 1) + off[m], ys, **_style(m, 'param'))
        top = max(top, max(ys))
    ax.set_ylim(0, top * 1.15 if top > 0 else 1)
    ax.set_xlim(0.6, K + 0.4)
    ax.set_xticks(range(1, K + 1))
    ax.set_xticklabels([f'S{k}' for k in range(1, K + 1)])
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    fmt = (lambda v, _: f'{v:g}') if top >= 10 else (lambda v, _: f'{v:.2g}')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt))


class Inches:
    """Place axes in absolute inches (y measured from the top), then convert."""

    def __init__(self, fig):
        self.fig = fig
        self.W, self.H = fig.get_size_inches()

    def axes(self, left, top, w, h):
        return self.fig.add_axes([left / self.W, 1 - (top + h) / self.H, w / self.W, h / self.H])

    def text(self, x, top, s, **kw):
        return self.fig.text(x / self.W, 1 - top / self.H, s, **kw)


# Block geometry (inches). One block = title, AUROC panel, two param panels.
YLAB = 0.42      # room for y tick labels + y label left of each panel
TITLE_H = 0.20
PERF_H = 1.42
PARAM_GAP = 0.30  # x tick labels of perf panel + param titles
PARAM_H = 0.62
TICK_H = 0.17
BLOCK_GAP = 0.10


def block_height(with_params=True):
    return TITLE_H + PERF_H + TICK_H + ((PARAM_GAP + PARAM_H + TICK_H) if with_params else 0)


def _fit_width(fig, text, max_in, min_size=5.6):
    """Shrink a text artist until it fits in max_in inches."""
    r = fig.canvas.get_renderer()
    while text.get_fontsize() > min_size:
        if text.get_window_extent(r).width / fig.dpi <= max_in:
            return
        text.set_fontsize(text.get_fontsize() - 0.2)


def draw_block(P, left, top, width, setup, methods, seeds, kind, metric='auroc',
               with_params=True):
    t = P.text(left + 0.06, top + 0.02,
               f'$\\mathbf{{Setup\\ {setup}}}$:  {SEQ_TITLE[setup]}', ha='left', va='top',
               fontsize=7.2, color=S.INK)
    _fit_width(P.fig, t, width - 0.08)
    ax = P.axes(left + YLAB, top + TITLE_H, width - YLAB, PERF_H)
    draw_perf(ax, setup, methods, seeds, metric)
    if not with_params:
        return
    pw = (width - YLAB - 0.34) / 2
    ptop = top + TITLE_H + PERF_H + PARAM_GAP
    enc_lbl, moe_lbl = PARAM_LABEL[kind]
    for i, (which, lbl) in enumerate([('enc', enc_lbl), ('moe', moe_lbl)]):
        pax = P.axes(left + YLAB + i * (pw + 0.34), ptop, pw, PARAM_H)
        draw_params(pax, setup, methods, which, kind)
        pax.set_title(lbl, fontsize=6.3, color=S.INK_2, pad=2.5)


# Compact ("side") geometry: parameter panels to the right of the AUROC panel,
# so Fig. 5 keeps roughly the height of the original figure (~4.4 in).
SIDE = dict(ylab=0.38, lab=0.42, pw=0.50, rtick=0.24, perf_h=1.55, title_h=0.20, tick_h=0.17)


def draw_block_side(P, left, top, width, setup, methods, seeds, kind, metric='auroc'):
    g = SIDE
    t = P.text(left + 0.06, top + 0.02,
               f'$\\mathbf{{Setup\\ {setup}}}$:  {SEQ_TITLE[setup]}', ha='left', va='top',
               fontsize=7.2, color=S.INK)
    _fit_width(P.fig, t, width - 0.08)
    aw = width - g['ylab'] - g['lab'] - g['pw'] - g['rtick']
    ax = P.axes(left + g['ylab'], top + g['title_h'], aw, g['perf_h'])
    draw_perf(ax, setup, methods, seeds, metric, labels_outside=True)
    px = left + g['ylab'] + aw + g['lab']
    ph = (g['perf_h'] - 0.10 - 0.24) / 2
    enc_lbl, moe_lbl = PARAM_LABEL_SHORT[kind]
    for i, (which, lbl) in enumerate([('enc', enc_lbl), ('moe', moe_lbl)]):
        pax = P.axes(px, top + g['title_h'] + 0.10 + i * (ph + 0.24), g['pw'], ph)
        draw_params(pax, setup, methods, which, kind)
        pax.yaxis.tick_right()
        pax.spines['left'].set_visible(False)
        pax.spines['right'].set_visible(True)
        pax.tick_params(axis='x', labelsize=5.8)
        pax.tick_params(axis='y', labelsize=5.8)
        if i == 0:
            pax.tick_params(axis='x', labelbottom=False)
        pax.set_title(lbl, fontsize=5.8, color=S.INK_2, pad=2)


def side_block_height():
    return SIDE['title_h'] + SIDE['perf_h'] + SIDE['tick_h']


def draw_legend(fig, methods, setups, y_frac):
    present = [t for t in S.TASK_ORDER if any(t in D.tasks(s) for s in setups)]
    mh = [Line2D([0], [0], **{k: v for k, v in _style(m, 'param').items() if k != 'zorder'},
                 label=_label(m)) for m in methods]
    for h in mh:
        h.set_linewidth(1.8 if h.get_label() == _label('ours') else 1.1)
    th = [Line2D([0], [0], color=S.INK_2, linestyle='none', marker=S.TASK_MARKER[t],
                 markersize=3.6, markerfacecolor=S.INK_2, label=D.TASK_LABEL[t]) for t in present]
    l1 = fig.legend(handles=mh, loc='center left', bbox_to_anchor=(0.01, y_frac),
                    ncol=2 if len(present) > 4 else len(mh), frameon=False, handlelength=3.0,
                    columnspacing=1.2, title='Method (line)', alignment='left')
    fig.legend(handles=th, loc='center right', bbox_to_anchor=(0.995, y_frac),
               ncol=4 if len(th) > 4 else len(th), frameon=False, handlelength=1.0,
               columnspacing=0.9, handletextpad=0.3, title='Task (marker)', alignment='left')
    fig.add_artist(l1)


def grid_figure(methods, setups, seeds_for, kind, metric='auroc', with_params=True,
                layout='stacked'):
    S.apply()
    n_rows = (len(setups) + 1) // 2
    bh = side_block_height() if layout == 'side' else block_height(with_params)
    leg_h = 0.46 if len(setups) % 2 == 0 else 0.0
    H = n_rows * bh + (n_rows - 1) * BLOCK_GAP + leg_h + 0.04
    fig = plt.figure(figsize=(S.TEXT_W, H))
    P = Inches(fig)
    bw = (S.TEXT_W - 0.12) / 2
    for i, s in enumerate(setups):
        r, c = divmod(i, 2)
        if layout == 'side':
            draw_block_side(P, c * (bw + 0.12), 0.02 + r * (bh + BLOCK_GAP), bw, s, methods,
                            seeds_for(s), kind, metric)
        else:
            draw_block(P, c * (bw + 0.12), 0.02 + r * (bh + BLOCK_GAP), bw, s, methods,
                       seeds_for(s), kind, metric, with_params)
    if len(setups) % 2:   # odd count: legend goes in the empty last slot
        P_top = 0.02 + (n_rows - 1) * (bh + BLOCK_GAP)
        _slot_legend(fig, methods, setups, x=(bw + 0.12 + YLAB) / S.TEXT_W,
                     y=1 - (P_top + bh * 0.45) / H)
    else:
        draw_legend(fig, methods, setups, (leg_h / 2) / H)
    return fig


def _slot_legend(fig, methods, setups, x, y):
    present = [t for t in S.TASK_ORDER if any(t in D.tasks(s) for s in setups)]
    mh = [Line2D([0], [0], **{k: v for k, v in _style(m, 'param').items() if k != 'zorder'},
                 label=_label(m)) for m in methods]
    th = [Line2D([0], [0], color=S.INK_2, linestyle='none', marker=S.TASK_MARKER[t],
                 markersize=3.8, markerfacecolor=S.INK_2, label=D.TASK_LABEL[t]) for t in present]
    l1 = fig.legend(handles=mh, loc='lower left', bbox_to_anchor=(x, y + 0.01), ncol=2,
                    frameon=False, handlelength=3.0, columnspacing=1.2, title='Method (line)',
                    alignment='left', fontsize=7, title_fontsize=7.5)
    fig.legend(handles=th, loc='upper left', bbox_to_anchor=(x, y - 0.01), ncol=3,
               frameon=False, handlelength=1.0, columnspacing=1.0, handletextpad=0.3,
               title='Task (marker)', alignment='left', fontsize=7, title_fontsize=7.5)
    fig.add_artist(l1)


def single_setup_figure(setup, methods, seeds, kind):
    S.apply()
    bw = 3.0
    n_tasks = len(D.tasks(setup))
    leg_h = 0.42 if n_tasks <= 4 else 0.56
    H = block_height() + leg_h + 0.04
    fig = plt.figure(figsize=(bw + 0.05, H))
    P = Inches(fig)
    draw_block(P, 0.0, 0.02, bw, setup, methods, seeds, kind)
    present = [t for t in S.TASK_ORDER if t in D.tasks(setup)]
    mh = [Line2D([0], [0], **{k: v for k, v in _style(m, 'param').items() if k != 'zorder'},
                 label=_label(m)) for m in methods]
    th = [Line2D([0], [0], color=S.INK_2, linestyle='none', marker=S.TASK_MARKER[t],
                 markersize=3.6, markerfacecolor=S.INK_2, label=D.TASK_LABEL[t]) for t in present]
    y1 = (leg_h * (0.72 if n_tasks <= 4 else 0.78)) / H
    y2 = (leg_h * (0.25 if n_tasks <= 4 else 0.30)) / H
    l1 = fig.legend(handles=mh, loc='center', bbox_to_anchor=(0.55, y1), ncol=len(mh),
                    frameon=False, handlelength=2.6, columnspacing=1.0)
    fig.legend(handles=th, loc='center', bbox_to_anchor=(0.55, y2),
               ncol=len(th) if len(th) <= 4 else 4,
               frameon=False, handlelength=1.0, columnspacing=0.8, handletextpad=0.3)
    fig.add_artist(l1)
    return fig


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'))
    plt.close(fig)
    print('wrote', os.path.join(OUT, name + '.{pdf,png}'))


def main():
    s42 = lambda s: [42]  # noqa: E731
    new_seeds = lambda s: D.SETUP4_TIMESTAMP_SEEDS if s == 4 else [42]  # noqa: E731
    # Fig. 5: LoRA is the MoE-only baseline (as in the original paper figures), so
    # its method-attributable encoder count would be 0 although it keeps the full
    # encoder. Inference-time totals are the like-for-like view -- and what the
    # original Fig. 5 / Fig. 9 plotted -- so they are the default.
    save(grid_figure(FIG5_METHODS, [1, 2, 3, 4], s42, 'inference', layout='side'),
         'fig5_cl_trajectories_compact')
    save(grid_figure(FIG5_METHODS, [1, 2, 3, 4], s42, 'inference'), 'fig5_cl_trajectories')
    save(grid_figure(FIG5_METHODS, [1, 2, 3, 4], s42, 'attributable'),
         'fig5_cl_trajectories_attributable')
    # The new baselines adapt only the MoE (their encoders freeze after a
    # modality's first stage), so their method-attributable encoder count is 0
    # although they keep the full, uncompressed encoder at inference. Inference
    # totals are the like-for-like comparison here, so they are the default.
    for kind, sfx in [('inference', ''), ('attributable', '_attributable')]:
        save(grid_figure(NEW_METHODS, [1, 2, 3, 4], new_seeds, kind),
             f'fig_cl_new_baselines{sfx}')
        save(grid_figure(NEW_METHODS, [1, 2, 3], new_seeds, kind),
             f'fig_cl_new_baselines_setups123{sfx}')
        for s in [1, 2, 3, 4]:
            save(single_setup_figure(s, NEW_METHODS, new_seeds(s), kind),
                 f'fig_cl_new_baselines_setup{s}{sfx}')
    save(grid_figure(FIG5_METHODS, [1, 2, 3, 4], s42, 'attributable', metric='auprc',
                     with_params=False), 'fig10_cl_auprc')
    # Figs. 8 and 9 redrawn from the same FLAME-CL runs as Fig. 5, so every
    # FLAME-CL line in the paper comes from one configuration. Lifelong-PT grows
    # the expert pool and adapts only the MoE, so inference totals are the
    # like-for-like view (attributable encoder count would be 0).
    save(grid_figure(['lifelong_pt', 'ours'], [1, 2, 3, 4], s42, 'inference'),
         'fig8_cl_vs_lifelong_pt')
    save(grid_figure(['lora', 'ours'], [1, 2, 3, 4], s42, 'inference'),
         'fig9_cl_vs_lora')
    # LoRA variants: the reported MoE-only baseline next to the encoder+MoE run
    # (rank-32 adapters on encoders and MoE; the run behind App. D.3.1's 109.54M -> 125.29M).
    LABEL_OVERRIDE.update({'lora': 'LoRA (MoE only)', 'lora_encmoe': 'LoRA (encoder+MoE)'})
    try:
        for kind, suffix in (('inference', ''), ('attributable', '_attributable')):
            save(grid_figure(['lora', 'lora_encmoe', 'ours'], [1, 2, 3, 4], s42, kind),
                 'fig_cl_lora_variants' + suffix)
    finally:
        LABEL_OVERRIDE.clear()


if __name__ == '__main__':
    main()
