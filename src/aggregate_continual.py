"""Generate LaTeX tables for continual-learning experiments.

Parses ``results_*.txt`` and ``param_count_method_attributable.txt`` from
hard-coded checkpoint dirs (one per (sequence, method) pair) and emits a single
LaTeX file with four major row sections:

    AUROC  |  AUPRC  |  Encoder params (M)  |  MoE params (M)

Each section is split by sequence ("CL setup"); within each sequence we list
the four CL methods (Simple FT / EWC / LoRA / FLAME-CL).

Columns are generic task positions (\\mathcal{A}, \\mathcal{B}, ...) up to the
longest sequence's task count. Each sequence's sub-section is preceded by a
"Tasks:" key row that maps the generic letters to actual task names so the
column meaning is unambiguous.

Performance values are the *final retention* for each task: the value pulled
from the LAST "After stage K" block in the results file (i.e. model after the
final stage trained, evaluated on every prior task).

Param values come from the per-stage rows of the param-count file and are
expressed in millions; tasks within the same training stage share the same
stage param value.
"""

import os
import re
from collections import OrderedDict


SEED = 42
BASE = '/cis/home/schaud35/clinical-highmmt/src/checkpoints/continual/laplace'
OUT_TEX = '/cis/home/schaud35/clinical-highmmt/src/results/continual_tables.tex'
OUT_PLOTS = '/cis/home/schaud35/clinical-highmmt/src/results/continual_plots'

TASK_TYPE = {
    'ihm': 'binary', 'los': 'binary', 'mortality': 'binary',
    'readmission': 'binary', 'risk': 'binary',
    'pheno': 'multilabel', 'birads': 'multiclass', 'density': 'multiclass',
}


def auroc_key(task):
    return 'auc' if TASK_TYPE.get(task) == 'binary' else 'ave_auc_macro'


def auprc_key(task):
    return 'auprc' if TASK_TYPE.get(task) == 'binary' else 'ave_auprc_macro'


def _cl_dir(method, wd='0.1'):
    """Map (method, wd) to the corresponding cl_*_... directory name."""
    common = ('_enc_all_trainable_target_moe_and_encoder_router_per_task'
              '_router_fixed_experts_rank32_replay0.0_alphaconst_0.0'
              f'_lr0.0001_wd{wd}_mod_drop_rate_0.0')
    if method == 'simple_ft':
        return f'cl_ewc_lamb0.0_alpha0.5_fi_true_no_router_exp{common}'
    if method == 'ewc':
        return f'cl_ewc_lamb1.0_alpha0.5_fi_true_no_router_exp{common}'
    if method == 'lora':
        return f'cl_lora_lorarank32{common}'
    if method == 'ours':
        return f'cl_ours{common}'
    raise ValueError(method)


def _full_path(seq_dir, mod_dir, cl_dir, fname):
    return os.path.join(BASE, seq_dir, mod_dir, str(SEED), cl_dir, fname)


def make_method_entry(seq_dir, mod_dir, method_key, wd, results_filename):
    cd = _cl_dir(method_key, wd)
    return {
        'results': _full_path(seq_dir, mod_dir, cd, results_filename),
        'params':  _full_path(seq_dir, mod_dir, cd, 'param_count_method_attributable.txt'),
    }


SEQUENCES = [
    {
        'id': 1,
        'name_tex': r'pheno $\to$ los $\to$ ihm',
        'name_plain': 'pheno → los → ihm',
        'seq_dir': 'pheno__los__ihm',
        'mod_dir': 'TS-Text-CXR_TS-Text-CXR_TS-Text-CXR',
        'stage_to_tasks': [['pheno'], ['los'], ['ihm']],
        'method_files': OrderedDict([
            ('Simple FT', make_method_entry('pheno__los__ihm', 'TS-Text-CXR_TS-Text-CXR_TS-Text-CXR', 'simple_ft', '0.1', 'results_full_rank.txt')),
            ('EWC',       make_method_entry('pheno__los__ihm', 'TS-Text-CXR_TS-Text-CXR_TS-Text-CXR', 'ewc',       '0.1', 'results_full_rank.txt')),
            ('LoRA',      make_method_entry('pheno__los__ihm', 'TS-Text-CXR_TS-Text-CXR_TS-Text-CXR', 'lora',      '0.1', 'results_full_rank.txt')),
            (r'\flamecl', make_method_entry('pheno__los__ihm', 'TS-Text-CXR_TS-Text-CXR_TS-Text-CXR', 'ours',      '1.0', 'results_reserved.txt')),
        ]),
    },
    {
        'id': 2,
        'name_tex': r'mortality $\to$ readmission',
        'name_plain': 'mortality → readmission',
        'seq_dir': 'mortality__readmission',
        'mod_dir': 'T1-T2-T3-T4-T5_T1-T2-T3-T4-T5',
        'stage_to_tasks': [['mortality'], ['readmission']],
        'method_files': OrderedDict([
            ('Simple FT', make_method_entry('mortality__readmission', 'T1-T2-T3-T4-T5_T1-T2-T3-T4-T5', 'simple_ft', '0.1', 'results_full_rank.txt')),
            ('EWC',       make_method_entry('mortality__readmission', 'T1-T2-T3-T4-T5_T1-T2-T3-T4-T5', 'ewc',       '0.1', 'results_full_rank.txt')),
            ('LoRA',      make_method_entry('mortality__readmission', 'T1-T2-T3-T4-T5_T1-T2-T3-T4-T5', 'lora',      '0.1', 'results_full_rank.txt')),
            (r'\flamecl', make_method_entry('mortality__readmission', 'T1-T2-T3-T4-T5_T1-T2-T3-T4-T5', 'ours',      '0.1', 'results_full_rank.txt')),
        ]),
    },
    {
        'id': 3,
        'name_tex': r'density $\to$ birads $\to$ risk',
        'name_plain': 'density → birads → risk',
        'seq_dir': 'density__birads__risk',
        'mod_dir': 'cc-mlo-2dcc-2dmlo_cc-mlo-2dcc-2dmlo_cc-mlo-2dcc-2dmlo',
        'stage_to_tasks': [['density'], ['birads'], ['risk']],
        'method_files': OrderedDict([
            ('Simple FT', make_method_entry('density__birads__risk', 'cc-mlo-2dcc-2dmlo_cc-mlo-2dcc-2dmlo_cc-mlo-2dcc-2dmlo', 'simple_ft', '0.1', 'results_full_rank.txt')),
            ('EWC',       make_method_entry('density__birads__risk', 'cc-mlo-2dcc-2dmlo_cc-mlo-2dcc-2dmlo_cc-mlo-2dcc-2dmlo', 'ewc',       '0.1', 'results_full_rank.txt')),
            ('LoRA',      make_method_entry('density__birads__risk', 'cc-mlo-2dcc-2dmlo_cc-mlo-2dcc-2dmlo_cc-mlo-2dcc-2dmlo', 'lora',      '0.1', 'results_full_rank.txt')),
            (r'\flamecl', make_method_entry('density__birads__risk', 'cc-mlo-2dcc-2dmlo_cc-mlo-2dcc-2dmlo_cc-mlo-2dcc-2dmlo', 'ours',      '1.0', 'results_full_rank.txt')),
        ]),
    },
    {
        'id': 4,
        'name_tex': r'pheno-density $\to$ los-birads-mortality $\to$ ihm-risk-readmission',
        'name_plain': 'pheno+density → los+birads+mortality → ihm+risk+readmission',
        'seq_dir': 'pheno-density__los-birads-mortality__ihm-risk-readmission',
        'mod_dir': 'TS-Text-CXR_cc-mlo-2dcc-2dmlo_TS-Text-CXR_cc-mlo-2dcc-2dmlo_T1-T2-T3-T4-T5_TS-Text-CXR_cc-mlo-2dcc-2dmlo_T1-T2-T3-T4-T5',
        'stage_to_tasks': [['pheno', 'density'], ['los', 'birads', 'mortality'], ['ihm', 'risk', 'readmission']],
        'method_files': OrderedDict([
            ('Simple FT', make_method_entry('pheno-density__los-birads-mortality__ihm-risk-readmission', 'TS-Text-CXR_cc-mlo-2dcc-2dmlo_TS-Text-CXR_cc-mlo-2dcc-2dmlo_T1-T2-T3-T4-T5_TS-Text-CXR_cc-mlo-2dcc-2dmlo_T1-T2-T3-T4-T5', 'simple_ft', '0.1', 'results_full_rank.txt')),
            ('EWC',       make_method_entry('pheno-density__los-birads-mortality__ihm-risk-readmission', 'TS-Text-CXR_cc-mlo-2dcc-2dmlo_TS-Text-CXR_cc-mlo-2dcc-2dmlo_T1-T2-T3-T4-T5_TS-Text-CXR_cc-mlo-2dcc-2dmlo_T1-T2-T3-T4-T5', 'ewc',       '0.1', 'results_full_rank.txt')),
            ('LoRA',      make_method_entry('pheno-density__los-birads-mortality__ihm-risk-readmission', 'TS-Text-CXR_cc-mlo-2dcc-2dmlo_TS-Text-CXR_cc-mlo-2dcc-2dmlo_T1-T2-T3-T4-T5_TS-Text-CXR_cc-mlo-2dcc-2dmlo_T1-T2-T3-T4-T5', 'lora',      '0.1', 'results_full_rank.txt')),
            (r'\flamecl', make_method_entry('pheno-density__los-birads-mortality__ihm-risk-readmission', 'TS-Text-CXR_cc-mlo-2dcc-2dmlo_TS-Text-CXR_cc-mlo-2dcc-2dmlo_T1-T2-T3-T4-T5_TS-Text-CXR_cc-mlo-2dcc-2dmlo_T1-T2-T3-T4-T5', 'ours',      '1.0', 'results_full_rank.txt')),
        ]),
    },
]


# --------------------------- parsing ---------------------------

_SECTION_HEADER_RE = re.compile(r'={3,}\s*\nAfter stage (\d+)\s*\(([^)]+)\)\s*--\s*\S+\s+eval\s*\n={3,}')
_SUB_HEADER_RE = re.compile(r'-{3,}\s*Stage (\d+)\s*\(([^)]+)\)\s*-{3,}')
_TASK_HEADER_RE = re.compile(r'-{3,}\s*Task (\d+)\s*\(([^)]+)\)\s*-{3,}')


def parse_results(filepath):
    """Return OrderedDict[after_stage_K (int)] -> OrderedDict[sub_stage_idx] -> {label, tasks: {task_name: {metric: float}}}."""
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        content = f.read()
    sections = OrderedDict()
    matches = list(_SECTION_HEADER_RE.finditer(content))
    for i, m in enumerate(matches):
        stage_k = int(m.group(1))
        body = content[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(content)]
        sub_matches = list(_SUB_HEADER_RE.finditer(body))
        sub_dict = OrderedDict()
        for j, sm in enumerate(sub_matches):
            sub_idx = int(sm.group(1))
            sub_label = sm.group(2).strip()
            sub_body = body[sm.end(): sub_matches[j + 1].start() if j + 1 < len(sub_matches) else len(body)]
            tasks = OrderedDict()
            tm_matches = list(_TASK_HEADER_RE.finditer(sub_body))
            for k, tm in enumerate(tm_matches):
                tname = tm.group(2).strip()
                tbody = sub_body[tm.end(): tm_matches[k + 1].start() if k + 1 < len(tm_matches) else len(sub_body)]
                tasks[tname] = _parse_metric_lines(tbody)
            sub_dict[sub_idx] = {'label': sub_label, 'tasks': tasks}
        sections[stage_k] = sub_dict
    return sections


def _parse_metric_lines(text):
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        k, _, v = line.partition(':')
        try:
            metrics[k.strip()] = float(v.strip())
        except ValueError:
            pass
    return metrics


_PARAM_ROW_RE = re.compile(r'^\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)')


def parse_params(filepath):
    """Return OrderedDict[stage_int] -> {label, total, moe, enc}."""
    if not os.path.exists(filepath):
        return None
    out = OrderedDict()
    with open(filepath) as f:
        for line in f:
            m = _PARAM_ROW_RE.match(line)
            if not m:
                continue
            out[int(m.group(1))] = {
                'label': m.group(2).strip(),
                'total': int(m.group(3).replace(',', '')),
                'moe':   int(m.group(4).replace(',', '')),
                'enc':   int(m.group(5).replace(',', '')),
            }
    return out


# --------------------------- value extraction ---------------------------

def final_retention(sections, task_name, metric_key):
    """Pull the metric for `task_name` from the LAST 'After stage K' block."""
    if not sections:
        return None
    last_k = max(sections.keys())
    for sub in sections[last_k].values():
        if task_name in sub['tasks']:
            return sub['tasks'][task_name].get(metric_key)
    return None


# --------------------------- LaTeX building ---------------------------

GREEK = ['$\\mathcal{A}$', '$\\mathcal{B}$', '$\\mathcal{C}$', '$\\mathcal{D}$',
        '$\\mathcal{E}$', '$\\mathcal{F}$', '$\\mathcal{G}$', '$\\mathcal{H}$']


def fmt_perf(v):
    return f"{v:.3f}" if v is not None else '--'


def fmt_param_m(v):
    return f"{v / 1e6:.2f}" if v is not None else '--'


def collect_perf_row(sequence, method_paths, metric):
    """Return list of strings, length len(flat_tasks)."""
    flat_tasks = [t for stage in sequence['stage_to_tasks'] for t in stage]
    sections = parse_results(method_paths['results'])
    cells = []
    for task in flat_tasks:
        key = auroc_key(task) if metric == 'auroc' else auprc_key(task)
        cells.append(fmt_perf(final_retention(sections, task, key)))
    return cells


def collect_param_row(sequence, method_paths, which):
    """Return list of strings, length = total tasks. Each task gets the param value of its stage."""
    params = parse_params(method_paths['params'])
    cells = []
    for stage_idx, tasks_in_stage in enumerate(sequence['stage_to_tasks']):
        if params and stage_idx in params:
            v = params[stage_idx].get(which)
            sval = fmt_param_m(v)
        else:
            sval = '--'
        for _ in tasks_in_stage:
            cells.append(sval)
    return cells


def build_table():
    n_cols = max(sum(len(s) for s in seq['stage_to_tasks']) for seq in SEQUENCES)
    method_col_w = 1
    col_spec = 'l' * method_col_w + 'c' * n_cols

    lines = [
        r'\begin{table*}[h!]',
        r'\centering',
        r'\small',
        r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{' + col_spec + '}',
        r'\toprule',
        'Method & ' + ' & '.join(GREEK[:n_cols]) + r' \\',
        r'\midrule',
    ]

    sections = [
        ('AUROC',                'auroc'),
        ('AUPRC',                'auprc'),
        (r'Encoder params (M)', 'enc'),
        (r'MoE params (M)',     'moe'),
    ]

    for sec_label, sec_key in sections:
        lines.append(f"\\multicolumn{{{n_cols + 1}}}{{l}}{{\\textbf{{{sec_label}}}}} \\\\")
        for seq in SEQUENCES:
            # Sequence sub-header
            lines.append(f"\\multicolumn{{{n_cols + 1}}}{{l}}{{\\quad \\emph{{Setup {seq['id']}: {seq['name_tex']}}}}} \\\\")
            # Task name key row
            flat_tasks = [t for stage in seq['stage_to_tasks'] for t in stage]
            task_cells = list(flat_tasks) + ['--'] * (n_cols - len(flat_tasks))
            task_cells_tex = [t.replace('_', r'\_') for t in task_cells]
            lines.append(r'\quad \textit{Tasks} & ' + ' & '.join(task_cells_tex) + r' \\')
            # One row per method
            for mname, paths in seq['method_files'].items():
                if sec_key in ('auroc', 'auprc'):
                    cells = collect_perf_row(seq, paths, sec_key)
                else:
                    cells = collect_param_row(seq, paths, sec_key)
                cells = cells + ['--'] * (n_cols - len(cells))
                lines.append(f"\\quad \\quad {mname} & " + ' & '.join(cells) + r' \\')
            lines.append(r'\addlinespace[2pt]')
        lines.append(r'\midrule')

    # Drop the trailing \midrule if any (we've added one too many)
    while lines and lines[-1].strip() in (r'\midrule', r'\addlinespace[2pt]'):
        lines.pop()

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        (r'\caption{Continual-learning results across four task sequences and four methods. '
         r'Performance rows (AUROC, AUPRC) report final retention: the metric on each task after '
         r'the model is trained through the entire stage sequence. Param rows give the '
         r'method-attributable parameter count (in millions) at the stage containing each task '
         r'(tasks within the same stage share the same value). Column letters $\mathcal{A}, \mathcal{B}, \dots$ '
         r'are positional within a sequence; the \emph{Tasks} row above each setup maps each '
         r'letter to a concrete task. AUROC uses \texttt{auc} for binary tasks and macro-averaged '
         r'AUROC for multilabel/multiclass tasks; AUPRC analogously.}'),
        r'\label{tab:continual-results}',
        r'\end{table*}',
    ]
    return '\n'.join(lines)


# --------------------------- plotting ---------------------------

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 — registers the 'science' style

# Use the SciencePlots aesthetic for paper-ready figures.
# 'no-latex' avoids requiring a working LaTeX install.
plt.style.use(['science', 'grid', 'no-latex'])

METHOD_ORDER = ['Simple FT', 'EWC', 'LoRA', r'\flamecl']
METHOD_DISPLAY = {'Simple FT': 'Simple FT', 'EWC': 'EWC', 'LoRA': 'LoRA', r'\flamecl': 'FLAME-CL'}
METHOD_COLOR = {
    'Simple FT': '#4C72B0',
    'EWC':       '#DD8452',
    'LoRA':      '#55A467',
    r'\flamecl': '#C44E52',
}

# Apply the user's preferred sizing for paper inclusion.
# These overrides apply to non-pure functions; the "pure SciencePlots" plot below
# resets them temporarily so SciencePlots' own defaults shine through.
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 25,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 21,
    'figure.titlesize': 26,
})


def _perf_for_method(sections, stage_train, task, metric_key):
    if not sections or stage_train not in sections:
        return None
    key = auroc_key(task) if metric_key == 'auroc' else auprc_key(task)
    for sub in sections[stage_train].values():
        if task in sub['tasks']:
            return sub['tasks'][task].get(key)
    return None


def _stage_xtick_labels(sequence):
    """Return list like ['stage 1\npheno', 'stage 2\nlos\nbirads\nmortality', ...].
    Each task in a multitask stage is stacked on its own line so labels never
    bleed sideways into the neighbouring subplot at large font sizes."""
    labels = []
    for i, tasks in enumerate(sequence['stage_to_tasks'], start=1):
        task_lines = '\n'.join(tasks)
        labels.append(f"stage {i}\n{task_lines}")
    return labels


TASK_LINESTYLES = [
    '-',
    '--',
    '-.',
    ':',
    (0, (5, 1)),
    (0, (1, 1)),
    (0, (3, 1, 1, 1)),
    (0, (5, 1, 1, 1, 1, 1)),
]
TASK_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

# Global task → (linestyle, marker) so the same task uses the same style
# across every subplot/figure (lets us show one shared legend).
GLOBAL_TASK_ORDER = ['pheno', 'los', 'ihm', 'mortality', 'readmission', 'density', 'birads', 'risk']
TASK_STYLE = {
    t: (TASK_LINESTYLES[i], TASK_MARKERS[i]) for i, t in enumerate(GLOBAL_TASK_ORDER)
}


def _task_style(task, fallback_idx):
    if task in TASK_STYLE:
        return TASK_STYLE[task]
    return (TASK_LINESTYLES[fallback_idx % len(TASK_LINESTYLES)],
            TASK_MARKERS[fallback_idx % len(TASK_MARKERS)])


def plot_perf(sequence, metric_key, metric_name, out_path):
    """Line plot: x = training stage, one line per (task, method).
    Color = method, linestyle+marker = task."""
    from matplotlib.lines import Line2D

    n_stages = len(sequence['stage_to_tasks'])
    method_paths = sequence['method_files']
    sections_per_method = {m: parse_results(p['results']) for m, p in method_paths.items()}

    flat_tasks = []
    task_first_stage = {}
    for stage_idx, tasks in enumerate(sequence['stage_to_tasks']):
        for t in tasks:
            flat_tasks.append(t)
            task_first_stage[t] = stage_idx + 1

    n_tasks = len(flat_tasks)
    # Wider canvas when there are many tasks so the right-hand legends don't crowd the axes.
    fig_w = 11.5 if n_tasks <= 4 else (12.5 if n_tasks <= 6 else 13.5)
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))

    all_values = []
    for t_i, task in enumerate(flat_tasks):
        ls, marker = _task_style(task, t_i)
        for method in METHOD_ORDER:
            sections = sections_per_method.get(method)
            xs, ys = [], []
            for stage_train in range(task_first_stage[task], n_stages + 1):
                v = _perf_for_method(sections, stage_train, task, metric_key)
                if v is not None:
                    xs.append(stage_train)
                    ys.append(v)
                    all_values.append(v)
            if xs:
                ax.plot(xs, ys,
                        color=METHOD_COLOR[method],
                        linestyle=ls,
                        marker=marker,
                        markersize=10,
                        linewidth=2.0,
                        alpha=0.88)

    ax.set_xticks(range(1, n_stages + 1))
    ax.set_xticklabels(_stage_xtick_labels(sequence))
    ax.set_xlabel("Training stage")
    ax.set_ylabel(metric_name)
    if all_values:
        vmin, vmax = min(all_values), max(all_values)
        span = max(vmax - vmin, 0.02)
        pad = max(0.015, span * 0.12)
        ax.set_ylim(max(0.0, vmin - pad), min(1.0, vmax + pad))
    else:
        ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_title(f"{metric_name}: {sequence['name_plain']}", pad=14)
    ax.margins(x=0.05)

    # Method legend (color key) and task legend (linestyle/marker key), both outside on the right.
    method_handles = [Line2D([0], [0], color=METHOD_COLOR[m], linewidth=3.0,
                              label=METHOD_DISPLAY[m]) for m in METHOD_ORDER]
    task_handles = []
    for i, task in enumerate(flat_tasks):
        ls, marker = _task_style(task, i)
        task_handles.append(Line2D([0], [0], color='gray', linestyle=ls, marker=marker,
                                   markersize=9, linewidth=2.0, label=task))

    leg_methods = ax.legend(handles=method_handles, title='Method',
                            loc='upper left', bbox_to_anchor=(1.02, 1.0),
                            borderaxespad=0.0, frameon=True,
                            fontsize=14, title_fontsize=15)
    ax.add_artist(leg_methods)
    leg_tasks = ax.legend(handles=task_handles, title='Task',
                          loc='upper left', bbox_to_anchor=(1.02, 0.55),
                          borderaxespad=0.0, frameon=True,
                          fontsize=14, title_fontsize=15)

    plt.savefig(out_path, bbox_inches='tight', dpi=150,
                bbox_extra_artists=[leg_methods, leg_tasks])
    plt.close(fig)


def plot_param(sequence, which, ylabel, out_path):
    """Line plot: x = training stage, one line per method (no task dim for params)."""
    n_stages = len(sequence['stage_to_tasks'])
    method_paths = sequence['method_files']
    params_per_method = {m: parse_params(p['params']) for m, p in method_paths.items()}

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for method in METHOD_ORDER:
        params = params_per_method.get(method)
        xs = list(range(1, n_stages + 1))
        ys = []
        for stage_idx in range(n_stages):
            if params and stage_idx in params:
                ys.append(params[stage_idx][which] / 1e6)
            else:
                ys.append(np.nan)
        ax.plot(xs, ys,
                color=METHOD_COLOR[method],
                marker='o', markersize=10,
                linewidth=2.4,
                label=METHOD_DISPLAY[method])

    ax.set_xticks(range(1, n_stages + 1))
    ax.set_xticklabels(_stage_xtick_labels(sequence))
    ax.set_xlabel("Training stage")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_title(f"{ylabel}: {sequence['name_plain']}", pad=14)
    ax.legend(loc='best', frameon=True)
    ax.margins(x=0.08)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_combined_grid(out_path):
    """3 (metric) x 4 (setup) grid.
    Rows = AUROC / Encoder params / MoE params; columns = Setup 1..4.
    Uniform subplot sizes, no per-subplot titles, shared legend at the bottom,
    row labels (metrics) on the left, column labels (setups) at the bottom."""
    from matplotlib.lines import Line2D

    row_specs = [
        ('auroc', 'AUROC'),
        ('enc',   'Encoder params (M)'),
        ('moe',   'MoE params (M)'),
    ]
    n_rows = len(row_specs)
    n_cols = len(SEQUENCES)

    # Pre-cache parsed files (one per setup).
    cache = []
    for seq in SEQUENCES:
        cache.append({
            'seq': seq,
            'sections': {m: parse_results(p['results']) for m, p in seq['method_files'].items()},
            'params':   {m: parse_params(p['params']) for m, p in seq['method_files'].items()},
        })

    fig = plt.figure(figsize=(26.0, 18.0))
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=0.075, right=0.985, top=0.93, bottom=0.16,
        wspace=0.30, hspace=0.18,
    )

    axes_grid = [[None] * n_cols for _ in range(n_rows)]

    for c, entry in enumerate(cache):
        seq = entry['seq']
        sections_per_method = entry['sections']
        params_per_method = entry['params']
        n_stages = len(seq['stage_to_tasks'])
        flat_tasks = [t for stage in seq['stage_to_tasks'] for t in stage]
        task_first_stage = {t: i + 1 for i, stage in enumerate(seq['stage_to_tasks']) for t in stage}

        for r, (metric_key, _name) in enumerate(row_specs):
            ax = fig.add_subplot(gs[r, c])
            axes_grid[r][c] = ax

            if metric_key == 'auroc':
                all_values = []
                for t_i, task in enumerate(flat_tasks):
                    ls, marker = _task_style(task, t_i)
                    for method in METHOD_ORDER:
                        sections = sections_per_method.get(method)
                        xs, ys = [], []
                        for stage_train in range(task_first_stage[task], n_stages + 1):
                            v = _perf_for_method(sections, stage_train, task, 'auroc')
                            if v is not None:
                                xs.append(stage_train)
                                ys.append(v)
                                all_values.append(v)
                        if xs:
                            ax.plot(xs, ys, color=METHOD_COLOR[method], linestyle=ls,
                                    marker=marker, markersize=9, linewidth=2.0, alpha=0.88)
                if all_values:
                    vmin, vmax = min(all_values), max(all_values)
                    span = max(vmax - vmin, 0.02)
                    pad = max(0.015, span * 0.12)
                    ax.set_ylim(max(0.0, vmin - pad), min(1.0, vmax + pad))
            else:
                for method in METHOD_ORDER:
                    params = params_per_method.get(method)
                    xs_p = list(range(1, n_stages + 1))
                    ys_p = []
                    for stage_idx in range(n_stages):
                        if params and stage_idx in params:
                            ys_p.append(params[stage_idx][metric_key] / 1e6)
                        else:
                            ys_p.append(np.nan)
                    ax.plot(xs_p, ys_p, color=METHOD_COLOR[method],
                            marker='o', markersize=10, linewidth=2.6)

            ax.set_xticks(range(1, n_stages + 1))
            if r == n_rows - 1:
                # Bottom row: show stage labels.
                ax.set_xticklabels(_stage_xtick_labels(seq), fontsize=20)
            else:
                # Upper rows: keep tick marks but hide labels.
                ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='y', labelsize=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            ax.margins(x=0.06)

    # Force a draw so positions are settled before placing fig.text labels.
    fig.canvas.draw()

    extra_artists = []

    # Row labels (metric names) on the left, vertically rotated.
    for r, (_key, name) in enumerate(row_specs):
        ax_left = axes_grid[r][0]
        bb = ax_left.get_position()
        y_center = (bb.y0 + bb.y1) / 2
        extra_artists.append(fig.text(
            0.018, y_center, name, ha='center', va='center',
            fontsize=30, fontweight='bold', rotation=90,
        ))

    # Column labels (setup id) sit at the TOP of the figure, above each column's
    # first subplot.
    for c, entry in enumerate(cache):
        ax_top = axes_grid[0][c]
        bb = ax_top.get_position()
        x_center = (bb.x0 + bb.x1) / 2
        extra_artists.append(fig.text(
            x_center, 0.965, f"Setup {entry['seq']['id']}",
            ha='center', va='bottom', fontsize=30, fontweight='bold',
        ))

    # Common legend: methods (color) + every unique task (linestyle/marker).
    method_handles = [Line2D([0], [0], color=METHOD_COLOR[m], linewidth=3.4,
                              label=METHOD_DISPLAY[m]) for m in METHOD_ORDER]
    task_handles = []
    for task in GLOBAL_TASK_ORDER:
        ls, marker = TASK_STYLE[task]
        task_handles.append(Line2D([0], [0], color='dimgray', linestyle=ls, marker=marker,
                                   markersize=10, linewidth=2.2, label=task))

    leg_methods = fig.legend(handles=method_handles, title='Method',
                             loc='lower left', bbox_to_anchor=(0.075, 0.005),
                             ncol=4, frameon=True,
                             fontsize=22, title_fontsize=23)
    fig.add_artist(leg_methods)
    leg_tasks = fig.legend(handles=task_handles, title='Task',
                           loc='lower right', bbox_to_anchor=(0.985, 0.005),
                           ncol=4, frameon=True,
                           fontsize=22, title_fontsize=23)

    fig.savefig(out_path, bbox_inches='tight', dpi=150,
                bbox_extra_artists=[leg_methods, leg_tasks] + extra_artists)
    plt.close(fig)


def plot_combined_grid_pure(out_path):
    """Same 3 (metric) x 4 (setup) layout but with minimal explicit overrides
    so the SciencePlots aesthetic dominates: thin lines, default cycler colors
    (re-mapped to methods so colors stay consistent), no manual font sizes,
    no manual linewidth/markersize. Uses a context-managed style stack to
    temporarily reset the user font-size overrides."""
    from matplotlib.lines import Line2D

    row_specs = [
        ('auroc', 'AUROC'),
        ('enc',   'Encoder params (M)'),
        ('moe',   'MoE params (M)'),
    ]
    n_rows = len(row_specs)
    n_cols = len(SEQUENCES)

    cache = []
    for seq in SEQUENCES:
        cache.append({
            'seq': seq,
            'sections': {m: parse_results(p['results']) for m, p in seq['method_files'].items()},
            'params':   {m: parse_params(p['params']) for m, p in seq['method_files'].items()},
        })

    # Apply only the SciencePlots style for this figure (drop the user's
    # font-size overrides applied at module import).
    with plt.style.context(['science', 'grid', 'no-latex']):
        cycle_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color',
                                                                     ['C0', 'C1', 'C2', 'C3'])
        method_color_local = {m: cycle_colors[i % len(cycle_colors)]
                              for i, m in enumerate(METHOD_ORDER)}

        fig = plt.figure(figsize=(20.0, 12.0))
        gs = fig.add_gridspec(
            n_rows, n_cols,
            left=0.06, right=0.985, top=0.97, bottom=0.18,
            wspace=0.30, hspace=0.55,
        )
        axes_grid = [[None] * n_cols for _ in range(n_rows)]

        for c, entry in enumerate(cache):
            seq = entry['seq']
            sections_per_method = entry['sections']
            params_per_method = entry['params']
            n_stages = len(seq['stage_to_tasks'])
            flat_tasks = [t for stage in seq['stage_to_tasks'] for t in stage]
            task_first_stage = {t: i + 1 for i, stage in enumerate(seq['stage_to_tasks'])
                                for t in stage}

            for r, (metric_key, _name) in enumerate(row_specs):
                ax = fig.add_subplot(gs[r, c])
                axes_grid[r][c] = ax

                if metric_key == 'auroc':
                    all_values = []
                    for t_i, task in enumerate(flat_tasks):
                        ls, marker = _task_style(task, t_i)
                        for method in METHOD_ORDER:
                            sections = sections_per_method.get(method)
                            xs, ys = [], []
                            for stage_train in range(task_first_stage[task], n_stages + 1):
                                v = _perf_for_method(sections, stage_train, task, 'auroc')
                                if v is not None:
                                    xs.append(stage_train)
                                    ys.append(v)
                                    all_values.append(v)
                            if xs:
                                # No explicit linewidth / markersize — SciencePlots defaults.
                                ax.plot(xs, ys,
                                        color=method_color_local[method],
                                        linestyle=ls, marker=marker)
                    if all_values:
                        vmin, vmax = min(all_values), max(all_values)
                        span = max(vmax - vmin, 0.02)
                        pad = max(0.015, span * 0.12)
                        ax.set_ylim(max(0.0, vmin - pad), min(1.0, vmax + pad))
                else:
                    for method in METHOD_ORDER:
                        params = params_per_method.get(method)
                        xs_p = list(range(1, n_stages + 1))
                        ys_p = []
                        for stage_idx in range(n_stages):
                            if params and stage_idx in params:
                                ys_p.append(params[stage_idx][metric_key] / 1e6)
                            else:
                                ys_p.append(np.nan)
                        ax.plot(xs_p, ys_p,
                                color=method_color_local[method], marker='o')

                ax.set_xticks(range(1, n_stages + 1))
                ax.set_xticklabels(_stage_xtick_labels(seq))
                ax.margins(x=0.06)

        fig.canvas.draw()

        extra_artists = []
        for r, (_key, name) in enumerate(row_specs):
            ax_left = axes_grid[r][0]
            bb = ax_left.get_position()
            y_center = (bb.y0 + bb.y1) / 2
            extra_artists.append(fig.text(
                0.013, y_center, name, ha='center', va='center', rotation=90,
            ))
        for c, entry in enumerate(cache):
            ax_bot = axes_grid[-1][c]
            bb = ax_bot.get_position()
            x_center = (bb.x0 + bb.x1) / 2
            extra_artists.append(fig.text(
                x_center, 0.115, f"Setup {entry['seq']['id']}",
                ha='center', va='top',
            ))

        method_handles = [Line2D([0], [0], color=method_color_local[m],
                                  label=METHOD_DISPLAY[m]) for m in METHOD_ORDER]
        task_handles = []
        for task in GLOBAL_TASK_ORDER:
            ls, marker = TASK_STYLE[task]
            task_handles.append(Line2D([0], [0], color='dimgray',
                                       linestyle=ls, marker=marker, label=task))

        leg_methods = fig.legend(handles=method_handles, title='Method',
                                 loc='lower left', bbox_to_anchor=(0.06, 0.005),
                                 ncol=4, frameon=True)
        fig.add_artist(leg_methods)
        leg_tasks = fig.legend(handles=task_handles, title='Task',
                               loc='lower right', bbox_to_anchor=(0.985, 0.005),
                               ncol=4, frameon=True)

        fig.savefig(out_path, bbox_inches='tight', dpi=150,
                    bbox_extra_artists=[leg_methods, leg_tasks] + extra_artists)
        plt.close(fig)


def render_all_plots():
    os.makedirs(OUT_PLOTS, exist_ok=True)
    written = []
    grid_png = os.path.join(OUT_PLOTS, 'continual_combined_grid.png')
    plot_combined_grid(grid_png)
    written.append(grid_png)
    for seq in SEQUENCES:
        for metric_key, metric_name in [('auroc', 'AUROC'), ('auprc', 'AUPRC')]:
            out = os.path.join(OUT_PLOTS, f"continual_setup{seq['id']}_{metric_name}.pdf")
            plot_perf(seq, metric_key, metric_name, out)
            written.append(out)
            out_png = out.replace('.pdf', '.png')
            plot_perf(seq, metric_key, metric_name, out_png)
            written.append(out_png)
        for which, ylabel, tag in [('enc', 'Encoder params (M)', 'encparams'),
                                    ('moe', 'MoE params (M)', 'moeparams')]:
            out = os.path.join(OUT_PLOTS, f"continual_setup{seq['id']}_{tag}.pdf")
            plot_param(seq, which, ylabel, out)
            written.append(out)
            out_png = out.replace('.pdf', '.png')
            plot_param(seq, which, ylabel, out_png)
            written.append(out_png)
    return written


def main():
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    tex = build_table()
    with open(OUT_TEX, 'w') as f:
        f.write(tex + '\n')
    print(f"Wrote: {OUT_TEX}")
    print()
    print("Rendering bar-chart trajectory plots...")
    paths = render_all_plots()
    for p in paths:
        print(f"Wrote: {p}")


if __name__ == '__main__':
    main()
