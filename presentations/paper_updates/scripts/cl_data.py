"""Shared loader for the continual-learning (CL) result files.

Reads the per-run ``results_full_rank.txt`` / ``param_count_*.txt`` files that
``src/continual_tasks.py`` writes under ``src/checkpoints/continual/laplace``.
The text parsers are imported from ``src/aggregate_continual.py`` (the paper's
original Fig. 5 registry) so the two never disagree on file format.

Run directories are chosen here, once, for every figure and table in
``presentations/paper_updates``:

* FLAME-CL  -> ``cl_ours_target_moe_and_encoder_..._wd1.0`` (encoder policy
  ``first_appearance``, the configuration described in Sec. 2.2 / App. E and
  the one whose forgetting is exactly 0 in Tab. 14). Setup 2 has no wd1.0 run,
  so it uses the wd0.1 run of the same configuration.
* Simple FT / EWC -> ``cl_ewc_lamb{0,1}..._enc_all_trainable_target_moe_and_encoder``
  (full encoder + MoE updated in place every stage, as App. D.3.1 describes).
* LoRA -> ``cl_lora_lorarank32_router_per_task_router_fixed_experts_rank32``:
  rank-32 adapters on the MoE experts only, the run the original paper figures
  (Fig. 9, the src/aggregate_continual.py registry) plotted and the baseline the
  authors intend. Its encoders follow first_appearance freezing, so its
  method-attributable encoder count is 0 although it keeps the full encoder at
  inference. NOTE: App. D.3.1's "LoRA grows from 109.54M to 125.29M" quotes the
  encoder+MoE run (``lora_encmoe``), which is NOT the reported baseline.
* LwF / SAME / TRGE / Lifelong-PT -> the same directories as the registry
  (SAME = the no-mask headline row; LwF = the valid run, not ``__INVALID_*``).

``results_full_rank.txt`` is used for every method: for FLAME-CL the value at a
task's own stage is *before* rank-32 truncation and every later value is the
deployed (truncated) model, so the one-time truncation cost stays visible --
the same convention as Tab. 13 ("first learned -> final").
"""

import os
import sys
from collections import OrderedDict

SRC = '/cis/home/schaud35/clinical-highmmt/src'
sys.path.insert(0, SRC)
import matplotlib  # noqa: E402
matplotlib.use('Agg')
from aggregate_continual import (  # noqa: E402  (also applies a global mpl style)
    parse_results, parse_params, parse_progression, auroc_key, auprc_key, TASK_TYPE,
)
import matplotlib.pyplot as plt  # noqa: E402
plt.rcdefaults()  # undo aggregate_continual's import-time style side effect

BASE = os.path.join(SRC, 'checkpoints/continual/laplace')
TAIL = '_replay0.0_alphaconst_0.0_lr0.0001_wd{wd}_mod_drop_rate_0.0'
FR = 'router_per_task_router_fixed_experts_rank32'
ENC_ALL = '_enc_all_trainable_target_moe_and_encoder_'

RUN_DIRS = {
    'ours':        'cl_ours_target_moe_and_encoder_' + FR + TAIL,
    'ours_all':    'cl_ours' + ENC_ALL + FR + TAIL,
    'ours_fullrank': 'cl_ours_fullrank_experts_fullrank_encoder_target_moe_and_encoder_' + FR + TAIL,
    'simple_ft':   'cl_ewc_lamb0.0_alpha0.5_fi_true_no_router_exp' + ENC_ALL + FR + TAIL,
    'ewc':         'cl_ewc_lamb1.0_alpha0.5_fi_true_no_router_exp' + ENC_ALL + FR + TAIL,
    'lora':        'cl_lora_lorarank32_' + FR + TAIL,          # MoE-only (reported)
    'lora_encmoe': 'cl_lora_lorarank32_target_moe_and_encoder_' + FR + TAIL,
    'lora_all':    'cl_lora_lorarank32' + ENC_ALL + FR + TAIL,
    'lwf':         'cl_lwf_lambda1.0_T2.0_' + FR + TAIL,
    'same':        'cl_same_mu0.9_tau0.1_er0.9_nomask_router_per_task_router_rank32' + TAIL,
    'trge':        'cl_trge_theta0.3_topk2_router_per_task_router_rank32' + TAIL,
    'lifelong_pt': 'cl_lifelong_pt_router_column_grow_rank16' + TAIL,
}
# Weight decay per (method). FLAME-CL runs at 1.0 except where only 0.1 exists.
DEFAULT_WD = {k: '0.1' for k in RUN_DIRS}
DEFAULT_WD.update({'ours': '1.0', 'ours_all': '1.0', 'ours_fullrank': '1.0'})

DISPLAY = {
    'ours': 'FLAME-CL', 'ours_all': 'FLAME-CL (enc. all-trainable)',
    'ours_fullrank': 'FLAME-CL (no truncation)',
    'simple_ft': 'Simple FT', 'ewc': 'EWC', 'lora': 'LoRA',
    'lora_all': 'LoRA (enc. all-trainable)', 'lora_encmoe': 'LoRA (encoder+MoE)',
    'lwf': 'LwF', 'same': 'SAME', 'trge': 'TRGE', 'lifelong_pt': 'Lifelong-PT',
}

SETUPS = OrderedDict([
    (1, dict(seq='pheno__los__ihm', mod='TS-Text-CXR_TS-Text-CXR_TS-Text-CXR',
             stages=[['pheno'], ['los'], ['ihm']])),
    (2, dict(seq='mortality__readmission', mod='T1-T2-T3-T4-T5_T1-T2-T3-T4-T5',
             stages=[['mortality'], ['readmission']], wd_override={'ours': '0.1'})),
    (3, dict(seq='density__birads__risk',
             mod='cc-mlo-2dcc-2dmlo_cc-mlo-2dcc-2dmlo_cc-mlo-2dcc-2dmlo',
             stages=[['density'], ['birads'], ['risk']])),
    (4, dict(seq='pheno-density__los-birads-mortality__ihm-risk-readmission',
             mod=('TS-Text-CXR_cc-mlo-2dcc-2dmlo_TS-Text-CXR_cc-mlo-2dcc-2dmlo_'
                  'T1-T2-T3-T4-T5_TS-Text-CXR_cc-mlo-2dcc-2dmlo_T1-T2-T3-T4-T5'),
             stages=[['pheno', 'density'], ['los', 'birads', 'mortality'],
                     ['ihm', 'risk', 'readmission']])),
])
SETUP4_TIMESTAMP_SEEDS = [703, 8622, 92784]

TASK_LABEL = {'pheno': 'PHENO', 'los': 'LOS', 'ihm': 'IHM', 'mortality': 'MOR',
              'readmission': 'RAD', 'density': 'DENSITY', 'birads': 'BIRADS',
              'risk': 'RISK'}


def run_dir(setup, method, seed=42):
    s = SETUPS[setup]
    wd = s.get('wd_override', {}).get(method, DEFAULT_WD[method])
    return os.path.join(BASE, s['seq'], s['mod'], str(seed),
                        RUN_DIRS[method].format(wd=wd))


def tasks(setup):
    return [t for st in SETUPS[setup]['stages'] for t in st]


def first_stage(setup):
    """task -> 1-based stage at which it is first trained."""
    return {t: i + 1 for i, st in enumerate(SETUPS[setup]['stages']) for t in st}


def trajectory(setup, method, seed=42, metric='auroc', which='full_rank'):
    """{task: [value after stage 1, ..., after stage K]} (None before first stage).

    which='full_rank' (default): FLAME-CL's own-stage value is pre-truncation.
    which='reserved': every value is the deployed (post-truncation) model; this
    is the reference Tab. 14 measures forgetting against."""
    path = os.path.join(run_dir(setup, method, seed), f'results_{which}.txt')
    sec = parse_results(path)
    if not sec:
        return None
    K = len(SETUPS[setup]['stages'])
    out = {}
    for t in tasks(setup):
        key = auroc_key(t) if metric == 'auroc' else auprc_key(t)
        row = []
        for k in range(1, K + 1):
            v = None
            for sub in sec.get(k, {}).values():
                if t in sub['tasks']:
                    v = sub['tasks'][t].get(key)
            row.append(v)
        out[t] = row
    return out


def params(setup, method, seed=42, kind='attributable'):
    """Per-stage parameter counts in millions: list of dicts {enc, moe}.

    kind='attributable': storage owned by the CL method (frozen low-rank stack
        for FLAME-CL, base + adapters for LoRA, the regularised weight set for
        Simple FT/EWC). This is the convention of the App. D.3.1 numbers.
    kind='inference': everything resident at inference (encoder; MoE experts
        + routers), from param_count_progression.txt.
    """
    d = run_dir(setup, method, seed)
    if kind == 'attributable':
        p = parse_params(os.path.join(d, 'param_count_method_attributable.txt'))
        if not p:
            return None
        return [{'enc': p[k]['enc'] / 1e6, 'moe': p[k]['moe'] / 1e6} for k in sorted(p)]
    p = parse_progression(os.path.join(d, 'param_count_progression.txt'))
    if not p:
        return None
    return [{'enc': p[k]['encoder'] / 1e6,
             'moe': (p[k]['moe_experts'] + p[k]['moe_routers']) / 1e6} for k in sorted(p)]


def cl_summary(setup, method, seed=42, metric='auroc'):
    """Per-task first-learned / final values plus AA, F, BWT for one run.

    first : value at the task's own stage from results_full_rank.txt (for
            FLAME-CL: before rank truncation -- the left side of Tab. 13).
    final : value after the last stage.
    F_i   : max_{s_i<=j<=K} R_{j,i} - R_{K,i} on the *deployed* model
            (results_reserved.txt), i.e. Tab. 14's convention: interference from
            later stages only, truncation cost excluded.
    BWT_i : R_{K,i} - R_{s_i,i} on the deployed model.
    AA    : mean final value over all tasks; F, BWT: mean over tasks learned
            before the last stage.
    """
    full = trajectory(setup, method, seed, metric, 'full_rank')
    dep = trajectory(setup, method, seed, metric, 'reserved') or full
    if full is None:
        return None
    K = len(SETUPS[setup]['stages'])
    fs = first_stage(setup)
    per = OrderedDict()
    for t in tasks(setup):
        s = fs[t] - 1
        d = [v for v in dep[t][s:] if v is not None]
        per[t] = dict(first=full[t][s], final=full[t][K - 1],
                      F=(max(d) - d[-1]) if fs[t] < K else None,
                      BWT=(d[-1] - d[0]) if fs[t] < K else None)
    prior = [t for t in tasks(setup) if fs[t] < K]
    return dict(per_task=per,
                AA=sum(per[t]['final'] for t in per) / len(per),
                F=sum(per[t]['F'] for t in prior) / len(prior),
                BWT=sum(per[t]['BWT'] for t in prior) / len(prior))


if __name__ == '__main__':
    for s in SETUPS:
        print(f'=== Setup {s}: ' + ' -> '.join('+'.join(st) for st in SETUPS[s]['stages']))
        for m in RUN_DIRS:
            tr = trajectory(s, m)
            if tr is None:
                print(f'  {m:12s} MISSING')
                continue
            cells = '  '.join(f"{TASK_LABEL[t]}:" + '/'.join('--' if v is None else f'{v:.3f}' for v in tr[t])
                              for t in tasks(s))
            print(f'  {m:12s} {cells}')
