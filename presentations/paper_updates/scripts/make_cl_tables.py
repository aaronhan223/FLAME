"""Camera-ready continual-learning tables (LaTeX + CSV + Markdown).

  tab_fig5_per_stage.tex          Per-stage numbers behind the redrawn Fig. 5
                                  (promised in the rebuttal, format of Table A):
                                  first learned -> final, AA, F, params.
  tab_new_baselines_setups123.tex FLAME-CL vs TRGE / SAME / LwF / EWC on Setups
                                  1-3 (Tab. 13/14 format, seed 42).
  tab13_tab14_setup4.tex          Tabs. 13 and 14 regenerated from the
                                  first_appearance FLAME-CL runs (3 timestamp seeds).
  tab_lora_variants_per_stage.tex FLAME-CL vs LoRA (MoE only, the reported baseline)
                                  vs LoRA (encoder+MoE), Setups 1-4, seed 42: first
                                  learned -> final, AA, F, params (both conventions).
  tab_lora_variants_summary.tex   compact main-text version: AA, F, encoder params.
  cl_trajectories_all.csv         every (setup, method, seed, task, stage) value.
  cl_tables.md                    Markdown copy of all of the above.

Drop convention (Tab. 13): a final value at least 0.02 below the first-learned
value is bold. F/BWT use the deployed model at the task's own stage as the
reference (post-truncation for FLAME-CL; Tab. 14).

    ~/anaconda3/envs/mmt/bin/python make_cl_tables.py
"""

import csv
import os
import sys
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cl_data as D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'tables')
DAT = os.path.join(ROOT, 'data')

SEQ_TEX = {
    1: r'\textsc{Pheno} $\to$ \textsc{Los} $\to$ \textsc{Ihm}',
    2: r'\textsc{Mor} $\to$ \textsc{Rad}',
    3: r'\textsc{Density} $\to$ \textsc{Birads} $\to$ \textsc{Risk}',
    4: r'\{\textsc{Pheno}, \textsc{Density}\} $\to$ \{\textsc{Los}, \textsc{Birads}, '
       r'\textsc{Mor}\} $\to$ \{\textsc{Ihm}, \textsc{Risk}, \textsc{Rad}\}',
}
SEQ_MD = {1: 'PHENO → LOS → IHM', 2: 'MOR → RAD', 3: 'DENSITY → BIRADS → RISK',
          4: '{PHENO, DENSITY} → {LOS, BIRADS, MOR} → {IHM, RISK, RAD}'}
TEX_NAME = {'ours': r'\textsc{FLAME-CL}', 'simple_ft': r'\textsc{Simple FT}', 'ewc': 'EWC',
            'lora': r'\textsc{LoRA}', 'trge': 'TRGE', 'same': 'SAME', 'lwf': r'L\textsc{w}F',
            'lora_encmoe': r'\textsc{LoRA} (enc.+MoE)'}
LORA_TEX = {'ours': r'\textsc{FLAME-CL}', 'lora': r'\textsc{LoRA} (MoE)',
            'lora_encmoe': r'\textsc{LoRA} (enc.+MoE)'}
LORA_MD = {'ours': 'FLAME-CL', 'lora': 'LoRA (MoE only)', 'lora_encmoe': 'LoRA (encoder+MoE)'}
DROP = 0.02


def _task_tex(t):
    return D.TASK_LABEL[t]


def _cell(first, final, own_last_stage, nd=3, std=None):
    """'first -> final' (bold final when it dropped by >= DROP)."""
    bold = (not own_last_stage) and first - final >= DROP - 1e-9

    def num(v, sd, b):
        body = f'{v:.{nd}f}'
        if sd is None:
            return r'\textbf{' + body + '}' if b else body
        body = r'\mathbf{' + body + '}' if b else body
        return f'${body}_{{\\pm {sd:.{nd}f}}}$'
    sa, sb = (std if std is not None else (None, None))
    if own_last_stage:
        return num(final, sb, False)
    return f'{num(first, sa, False)} $\\to$ {num(final, sb, bold)}'


def _md_cell(first, final, own_last_stage, nd=3, std=None):
    s = (lambda v, sd: f'{v:.{nd}f}' + (f'±{sd:.{nd}f}' if sd is not None else ''))
    sa = std[0] if std else None
    sb = std[1] if std else None
    if own_last_stage:
        return s(final, sb)
    b = s(final, sb)
    if first - final >= DROP - 1e-9:
        b = f'**{b}**'
    return f'{s(first, sa)} → {b}'


# ---------------------------------------------------------------- single seed

def setup_block(setup, methods, param_kind, seed=42):
    """LaTeX rows + Markdown rows for one setup at one seed."""
    K = len(D.SETUPS[setup]['stages'])
    fs = D.first_stage(setup)
    summ = {m: D.cl_summary(setup, m, seed) for m in methods}
    n = len(methods)
    tex = [r'\midrule',
           rf'\multicolumn{{{n + 1}}}{{l}}{{\emph{{Setup {setup}:}} {SEQ_TEX[setup]}}} \\',
           r'\addlinespace[1pt]']
    md = [f'| **Setup {setup}: {SEQ_MD[setup]}** |' + ' |' * n]
    for t in D.tasks(setup):
        last = fs[t] == K
        tc, mc = [], []
        for m in methods:
            p = summ[m]['per_task'][t]
            tc.append(_cell(p['first'], p['final'], last))
            mc.append(_md_cell(p['first'], p['final'], last))
        tex.append(f'{_task_tex(t)} (s{fs[t]}) & ' + ' & '.join(tc) + r' \\')
        md.append(f'| {D.TASK_LABEL[t]} (s{fs[t]}) | ' + ' | '.join(mc) + ' |')
    best_aa = max(summ[m]['AA'] for m in methods)
    aa = [(r'\textbf{%.3f}' if abs(summ[m]['AA'] - best_aa) < 5e-4 else '%.3f') % summ[m]['AA']
          for m in methods]
    tex += [r'\addlinespace[1pt]',
            'AA $\\uparrow$ & ' + ' & '.join(aa) + r' \\',
            'F $\\downarrow$ & ' + ' & '.join(f"{summ[m]['F']:.3f}" for m in methods) + r' \\']
    md += ['| AA ↑ | ' + ' | '.join(f"{summ[m]['AA']:.3f}" for m in methods) + ' |',
           '| F ↓ | ' + ' | '.join(f"{summ[m]['F']:.3f}" for m in methods) + ' |']

    kinds = (param_kind,) if isinstance(param_kind, str) else tuple(param_kind)
    for kind in kinds:
        prm = {m: D.params(setup, m, seed, kind) for m in methods}

        def _p(m):
            if prm[m] is None:
                return '--', '--'
            a, b = prm[m][0], prm[m][K - 1]
            fmt = (lambda v: f'{v:.1f}') if max(a['enc'], b['enc']) >= 10 else (lambda v: f'{v:.2f}')
            enc = fmt(a['enc']) if abs(a['enc'] - b['enc']) < 1e-9 else f"{fmt(a['enc'])}$\\to${fmt(b['enc'])}"
            moe = f"{a['moe']:.2f}" if abs(a['moe'] - b['moe']) < 1e-9 else f"{a['moe']:.2f}$\\to${b['moe']:.2f}"
            return enc, moe
        lbl = 'method-attr.' if kind == 'attributable' else 'at inference'
        tex.append(f'Enc.\\ params, {lbl} (M) & ' + ' & '.join(_p(m)[0] for m in methods) + r' \\')
        tex.append(f'MoE params, {lbl} (M) & ' + ' & '.join(_p(m)[1] for m in methods) + r' \\')
        md.append(f'| Enc. params, {lbl} (M) | ' + ' | '.join(_p(m)[0].replace('$\\to$', '→') for m in methods) + ' |')
        md.append(f'| MoE params, {lbl} (M) | ' + ' | '.join(_p(m)[1].replace('$\\to$', '→') for m in methods) + ' |')
    return tex, md


def single_seed_table(setups, methods, param_kind, caption, label, tex_names=None, md_names=None):
    n = len(methods)
    tex_names, md_names = tex_names or TEX_NAME, md_names or D.DISPLAY
    tex = [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3.5pt}',
           rf'\caption{{{caption}}}', rf'\label{{{label}}}',
           r'\begin{tabular}{l' + 'c' * n + '}', r'\toprule',
           'Task (stage) & ' + ' & '.join(tex_names[m] for m in methods) + r' \\']
    md = ['| Task (stage) | ' + ' | '.join(md_names[m] for m in methods) + ' |',
          '|---' * (n + 1) + '|']
    for s in setups:
        t, m_ = setup_block(s, methods, param_kind)
        tex += t
        md += m_
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(tex), '\n'.join(md)


def lora_variants_summary(methods=('ours', 'lora', 'lora_encmoe'), seed=42):
    """Main-text table: AA, F and final-stage encoder parameters per setup."""
    short = {'ours': r'\textsc{FLAME}', 'lora': r'MoE', 'lora_encmoe': r'enc.+MoE'}
    n = len(methods)
    groups = ['AA $\\uparrow$', 'F $\\downarrow$', 'Enc.\\ params, inference (M)',
              'Enc.\\ params, method-attr.\\ (M)']
    tex = [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{2.0pt}',
           r'\caption{\textsc{FLAME-CL} against \textsc{LoRA} with rank-32 adapters on the MoE experts '
           r'only (MoE; the baseline of Fig.~\ref{fig:cl}) and on the encoders and the MoE (enc.+MoE), '
           r'seed 42. AA and F over all tasks after the final stage; encoder parameters after the final '
           r'stage, as the total resident at inference and as the part attributable to the CL method '
           r'(\textsc{FLAME-CL}: its frozen low-rank stack; \textsc{LoRA}: base encoder plus adapters it '
           r'trains, 0 for MoE-only adapters). Per-task numbers in Tab.~\ref{tab:cl-lora-variants}.}',
           r'\label{tab:cl-lora-variants-summary}',
           r'\begin{tabular}{@{}l' + 'c' * (n * len(groups)) + r'@{}}', r'\toprule',
           ' & ' + ' & '.join(rf'\multicolumn{{{n}}}{{c}}{{{g}}}' for g in groups) + r' \\',
           ''.join(rf'\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}' for i in range(len(groups))),
           'Setup & ' + ' & '.join([' & '.join(short[m] for m in methods)] * len(groups)) + r' \\',
           r'\midrule']
    md = ['| Setup | ' + ' | '.join(f'{g} {LORA_MD[m]}' for g in ['AA', 'F', 'enc. inference (M)',
                                                                    'enc. method-attr. (M)']
                                     for m in methods) + ' |',
          '|---' * (1 + n * len(groups)) + '|']
    for s in D.SETUPS:
        K = len(D.SETUPS[s]['stages'])
        summ = {m: D.cl_summary(s, m, seed) for m in methods}
        inf = {m: D.params(s, m, seed, 'inference')[K - 1]['enc'] for m in methods}
        att = {m: D.params(s, m, seed, 'attributable')[K - 1]['enc'] for m in methods}
        best = max(summ[m]['AA'] for m in methods)
        aa = [(r'\textbf{%.3f}' if abs(summ[m]['AA'] - best) < 5e-4 else '%.3f') % summ[m]['AA']
              for m in methods]
        fmt = lambda v: f'{v:.1f}' if v >= 10 else f'{v:.2f}'  # noqa: E731
        cells = aa + [f"{summ[m]['F']:.3f}" for m in methods] + [fmt(inf[m]) for m in methods] \
            + [fmt(att[m]) for m in methods]
        tex.append(f'{s} & ' + ' & '.join(cells) + r' \\')
        md.append(f'| {s} | ' + ' | '.join(c.replace(r'\textbf{', '**').replace('}', '**') for c in cells) + ' |')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(tex), '\n'.join(md)


# ---------------------------------------------------------------- three seeds

def setup4_multiseed(methods, seeds):
    K = 3
    fs = D.first_stage(4)
    runs = {m: [D.cl_summary(4, m, s) for s in seeds] for m in methods}
    for m, r in runs.items():
        if any(x is None for x in r):
            raise RuntimeError(f'missing Setup 4 seed for {m}')
    ms = lambda xs: (mean(xs), stdev(xs) if len(xs) > 1 else 0.0)  # noqa: E731

    # --- Tab. 13: first -> final
    t13 = [r'\begin{table}[t]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{1.8pt}',
           r'\caption{Setup 4 per-task AUROC when first learned $\to$ after the final stage '
           r'(3 timestamp-generated seeds, mean $\pm$ std). Stage-3 tasks are learned last and '
           r'report one value. Drops of at least 0.02 in bold.}',
           r'\label{tab:cl-setup4-retention}',
           r'\begin{tabular}{l' + 'c' * len(methods) + '}', r'\toprule',
           'Task (stage) & ' + ' & '.join(TEX_NAME[m] for m in methods) + r' \\', r'\midrule']
    md13 = ['| Task (stage) | ' + ' | '.join(D.DISPLAY[m] for m in methods) + ' |',
            '|---' * (len(methods) + 1) + '|']
    for t in D.tasks(4):
        last = fs[t] == K
        tc, mc = [], []
        for m in methods:
            a = ms([r['per_task'][t]['first'] for r in runs[m]])
            b = ms([r['per_task'][t]['final'] for r in runs[m]])
            tc.append(_cell(a[0], b[0], last, 2, (a[1], b[1])))
            mc.append(_md_cell(a[0], b[0], last, 2, (a[1], b[1])))
        t13.append(f'{_task_tex(t)} (s{fs[t]}) & ' + ' & '.join(tc) + r' \\')
        md13.append(f'| {D.TASK_LABEL[t]} (s{fs[t]}) | ' + ' | '.join(mc) + ' |')
    t13 += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    # --- Tab. 14: forgetting per task + summary
    t14 = [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
           r'\caption{Setup 4 per-task forgetting F and summary metrics (3 timestamp-generated '
           r'seeds, mean $\pm$ std). F is measured against the model deployed at the stage where each '
           r'task was learned (post-truncation for \textsc{FLAME-CL}).}',
           r'\label{tab:cl-setup4-forgetting}',
           r'\begin{tabular}{l' + 'c' * len(methods) + '}', r'\toprule',
           ' & ' + ' & '.join(TEX_NAME[m] for m in methods) + r' \\', r'\midrule']
    md14 = ['| | ' + ' | '.join(D.DISPLAY[m] for m in methods) + ' |',
            '|---' * (len(methods) + 1) + '|']
    for t in [t for t in D.tasks(4) if fs[t] < K]:
        vals = [ms([r['per_task'][t]['F'] for r in runs[m]]) for m in methods]
        t14.append(f'F: {_task_tex(t)} (s{fs[t]}) & ' +
                   ' & '.join(f'${v:.3f}\\pm{s:.3f}$' for v, s in vals) + r' \\')
        md14.append(f'| F: {D.TASK_LABEL[t]} (s{fs[t]}) | ' +
                    ' | '.join(f'{v:.3f}±{s:.3f}' for v, s in vals) + ' |')
    t14.append(r'\midrule')
    for key, lbl in [('F', r'F (avg) $\downarrow$'), ('BWT', r'BWT (avg) $\uparrow$'),
                     ('AA', r'AA (8 tasks) $\uparrow$')]:
        vals = [ms([r[key] for r in runs[m]]) for m in methods]
        t14.append(f'{lbl} & ' + ' & '.join(f'${v:.3f}\\pm{s:.3f}$' for v, s in vals) + r' \\')
        md14.append(f"| {lbl.split(' $')[0]} | " +
                    ' | '.join(f'{v:.3f}±{s:.3f}' for v, s in vals) + ' |')
    t14 += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(t13), '\n'.join(t14), '\n'.join(md13), '\n'.join(md14)


# ---------------------------------------------------------------- CSV dump

def dump_csv(path):
    rows = []
    jobs = [(s, m, 42) for s in D.SETUPS for m in
            ['simple_ft', 'ewc', 'lora', 'lora_encmoe', 'lwf', 'same', 'trge', 'lifelong_pt', 'ours']]
    jobs += [(4, m, sd) for m in ['ewc', 'lwf', 'same', 'trge', 'ours']
             for sd in D.SETUP4_TIMESTAMP_SEEDS]
    for s, m, sd in jobs:
        for metric in ('auroc', 'auprc'):
            tr = D.trajectory(s, m, sd, metric)
            if tr is None:
                continue
            for t, vals in tr.items():
                for k, v in enumerate(vals, 1):
                    if v is not None:
                        rows.append([s, D.DISPLAY[m], D.RUN_DIRS[m].split('_replay')[0], sd,
                                     D.TASK_LABEL[t], D.first_stage(s)[t], k, metric, f'{v:.6f}'])
        for kind in ('attributable', 'inference'):
            p = D.params(s, m, sd, kind)
            if p:
                for k, row in enumerate(p, 1):
                    rows.append([s, D.DISPLAY[m], D.RUN_DIRS[m].split('_replay')[0], sd, '', '', k,
                                 f'enc_params_M_{kind}', f"{row['enc']:.6f}"])
                    rows.append([s, D.DISPLAY[m], D.RUN_DIRS[m].split('_replay')[0], sd, '', '', k,
                                 f'moe_params_M_{kind}', f"{row['moe']:.6f}"])
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['setup', 'method', 'run_dir_prefix', 'seed', 'task', 'task_first_stage',
                    'after_stage', 'quantity', 'value'])
        w.writerows(rows)
    print('wrote', path, f'({len(rows)} rows)')


def main():
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(DAT, exist_ok=True)
    md = ['# Continual-learning tables (generated by make_cl_tables.py)', '']

    tex, m = single_seed_table(
        [1, 2, 3, 4], ['simple_ft', 'ewc', 'lora', 'ours'], 'inference',
        r'Per-stage numbers behind Fig.~\ref{fig:cl}: AUROC when first learned $\to$ after the '
        r'final stage (seed 42). Stage-$K$ tasks report one value; drops of at least 0.02 in '
        r'bold. For \textsc{FLAME-CL} the first value is before rank-32 truncation, so its '
        r'deltas are the one-time truncation cost; F is measured against the deployed '
        r'(post-truncation) model. Parameter rows give inference-time totals after the first $\to$ '
        r'last stage; \textsc{LoRA} adapts only the MoE and keeps the full encoder.',
        'tab:cl-fig5-per-stage')
    open(os.path.join(TAB, 'tab_fig5_per_stage.tex'), 'w').write(tex + '\n')
    md += ['## Per-stage numbers behind Fig. 5 (seed 42)', '', m, '']

    tex, m = single_seed_table(
        [1, 2, 3], ['ours', 'trge', 'same', 'lwf', 'ewc'], 'inference',
        r'Setups 1--3: AUROC when first learned $\to$ after the final stage for '
        r'\textsc{FLAME-CL} and the MoE-based / distillation baselines (seed 42; Setup 4 is in '
        r'Tabs.~\ref{tab:cl-setup4-retention},~\ref{tab:cl-setup4-forgetting}). Drops of at least '
        r'0.02 in bold; F against the model deployed at the stage where each task was learned. Parameter rows '
        r'give inference-time totals after the first $\to$ last stage: TRGE, SAME and L\textsc{w}F '
        r'adapt only the MoE and keep the full encoder.',
        'tab:cl-new-baselines-s123')
    open(os.path.join(TAB, 'tab_new_baselines_setups123.tex'), 'w').write(tex + '\n')
    md += ['## New baselines, Setups 1–3 (seed 42)', '', m, '']

    t13, t14, m13, m14 = setup4_multiseed(['ours', 'trge', 'same', 'lwf', 'ewc'],
                                          D.SETUP4_TIMESTAMP_SEEDS)
    open(os.path.join(TAB, 'tab13_tab14_setup4.tex'), 'w').write(t13 + '\n\n' + t14 + '\n')
    md += ['## Setup 4, 3 timestamp seeds — Tab. 13 (first learned → final)', '', m13, '',
           '## Setup 4, 3 timestamp seeds — Tab. 14 (forgetting, AA, BWT)', '', m14, '']

    tex, m = single_seed_table(
        [1, 2, 3, 4], ['ours', 'lora', 'lora_encmoe'], ('inference', 'attributable'),
        r'\textsc{FLAME-CL} against the two \textsc{LoRA} variants (rank-32 adapters on the MoE '
        r'experts only, the baseline of Fig.~\ref{fig:cl}; and on the encoders and the MoE): AUROC '
        r'when first learned $\to$ after the final stage (seed 42). Stage-$K$ tasks report one value; '
        r'drops of at least 0.02 in bold. For \textsc{FLAME-CL} the first value is before rank-32 '
        r'truncation. Parameters after the first $\to$ last stage, as inference-time totals and as '
        r'method-attributable counts.',
        'tab:cl-lora-variants', tex_names=LORA_TEX, md_names=LORA_MD)
    open(os.path.join(TAB, 'tab_lora_variants_per_stage.tex'), 'w').write(tex + '\n')
    md += ['## FLAME-CL vs the two LoRA variants (seed 42)', '', m, '']
    tex, m = lora_variants_summary()
    open(os.path.join(TAB, 'tab_lora_variants_summary.tex'), 'w').write(tex + '\n')
    md += ['## FLAME-CL vs the two LoRA variants, summary (seed 42)', '', m, '']

    open(os.path.join(TAB, 'cl_tables.md'), 'w').write('\n'.join(md))
    print('wrote', TAB)
    dump_csv(os.path.join(DAT, 'cl_trajectories_all.csv'))


if __name__ == '__main__':
    main()
