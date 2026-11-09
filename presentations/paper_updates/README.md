# Camera-ready updates promised in the rebuttal

Everything here is generated from result files and checkpoints already on io79; no new training was run.
The only new GPU work is evaluation-only collection on the saved FLAME-CL checkpoints for the alignment figure (§1, last row).
Open `preview/preview.pdf` first: it shows every figure and table at NeurIPS text width (5.5 in).

```bash
# on your laptop (io79 is the source of presentations/; it is not on GitHub except for this folder)
rsync -avz --update --exclude=.DS_Store --exclude=__pycache__ \
  io79:/cis/home/schaud35/clinical-highmmt/presentations/ presentations/
# on io79, to rebuild everything
bash presentations/paper_updates/make_all.sh
```

## 1. What to drop into the paper

| Rebuttal promise / request | File(s) | Goes where |
|---|---|---|
| Redraw Fig. 5: legend agrees with the text (LoRA, not Lifelong-PT); FLAME-CL on a distinct line style; setups on two rows | `figs/fig5_cl_trajectories_compact.pdf` (main text, about the height of the old Fig. 5); `figs/fig5_cl_trajectories.pdf` (taller variant, parameter panels underneath) | replaces Fig. 5 |
| Per-stage numbers for Fig. 5 (Table A format) | `tables/tab_fig5_per_stage.tex` | new appendix table, App. D.3.1 |
| AUPRC companion of Fig. 5 (it had the same Lifelong-PT legend) | `figs/fig10_cl_auprc.pdf` | replaces Fig. 10 |
| Per-setup plots for the new baselines (SAME, TRGE, LwF), Setups 1–3 | `figs/fig_cl_new_baselines_setups123.pdf`, or one file per setup `figs/fig_cl_new_baselines_setup{1,2,3}.pdf`; `figs/fig_cl_new_baselines.pdf` adds Setup 4 (3 seeds) | new figure, App. D.3.2 |
| Numbers for those plots | `tables/tab_new_baselines_setups123.tex` | App. D.3.2, next to Tab. 13 |
| Per-task loss-weight tables | `tables/tab_aux_loss_per_task.tex` (two tables) | App. D.2.5, after Tab. 12 |
| Spearman ρ for the robustness section | `tables/robustness_spearman.tex` + drop-in sentences `tables/robustness_d6_snippet.tex` | App. D.6 |
| Page 9 has too many tables | `figs/fig_cl_summary_tab3_tab4.pdf` (Tab. 3 as a dot/bar chart + Tab. 4 as a heatmap) | replaces Tabs. 3 and 4 (exact values stay in Tabs. 14, 16) |
| (extra) Tabs. 13/14 regenerated from the same runs | `tables/tab13_tab14_setup4.tex` | replaces Tabs. 13, 14 |
| (extra) Figs. 8, 9 from the same FLAME-CL runs as Fig. 5 | `figs/fig8_cl_vs_lifelong_pt.pdf`, `figs/fig9_cl_vs_lora.pdf` | replace Figs. 8, 9 (Fig. 9 is a subset of Fig. 5; dropping it is suggested) |
| FLAME-CL vs **both LoRA variants**: MoE-only (the reported baseline) and encoder+MoE | `figs/fig_cl_lora_variants.pdf` (Fig. 9 format, 3 methods; `_attributable` twin); main text: `tables/tab_lora_variants_summary.tex` (AA, F, encoder params per setup); appendix: `tables/tab_lora_variants_per_stage.tex` (per task, both parameter conventions) | main text next to RQ3/RQ4 or App. D.3.1; the figure can replace Fig. 9 |
| Continual-phase spectra **and singular-vector alignment across all four sequences** (to 9k3u) | **Path A (recommended):** `figs/fig_cl_spectra_alignment.pdf` (4 setups × {input, weight-only, data-aware, alignment}) + `tables/tab_cl_spectra_alignment.tex`. **Path B:** keep Fig. 16 and add `figs/fig16_setup{1,2,3}_stage*.png` (same format as Fig. 16) + `figs/fig_cl_alignment.pdf` (alignment only) + the same table. LaTeX for both, plus the new D.4.3 paragraph: `tables/cl_alignment_snippets.tex` | App. D.4.3; path A replaces Fig. 16 |

**Formats.** PNG twins of every figure and the raw values in `data/*.csv` exist on io79 (rsync them, or rerun `make_all.sh`).
They are not pushed, because the repo `.gitignore` excludes `*.png` and `*.csv`. Markdown copies of all tables are in `tables/*.md`.

**LaTeX.** The `.tex` tables need only `booktabs`. Their `\label`s are new (`tab:cl-fig5-per-stage`, `tab:cl-new-baselines-s123`,
`tab:cl-setup4-retention`, `tab:cl-setup4-forgetting`, `tab:aux-bal-per-task`, `tab:aux-int-per-task`, `tab:robustness-spearman`,
`tab:cl-spectra-alignment`, `fig:cl-spectra-alignment`, `fig:cl-alignment`, `tab:cl-lora-variants`, `tab:cl-lora-variants-summary`). Path B's Setup 1–3 pages use `subfigure`, as Fig. 16 does (`subcaption`).
The robustness caption refers to `app:robustness`: rename that reference to App. D.6's label.

**Variants.** `*_attributable.pdf` twins plot method-attributable parameter counts instead of inference-time totals (see §2).

### Suggested captions

**Fig. 5.** Per-stage continual-learning trajectories for Simple FT, EWC, LoRA (MoE adapters) and FLAME-CL on the four task sequences
(seed 42). For each setup, the large panel is the AUROC of every task seen so far after each stage S*k* (marker = task, colour and line
style = method, FLAME-CL solid); the small panels give encoder and MoE parameters at inference. For FLAME-CL the value at a task's own
stage is before rank-32 truncation, so its only drops are the one-time truncation cost. Numbers in Tab. X; AUPRC in Fig. 10.

**Tabs. 3–4 figure.** (a) Setup 4, 3 timestamp seeds: average accuracy (AA, dot ± std), forgetting F (bar ± std) and parameters after
three stages for FLAME-CL, the baselines and the component ablations (hollow/hatched). (b) CL-phase rank sweep (Setup 4, seed 42):
AUROC when the active component of a task's own stage is truncated to rank r, earlier stages frozen at rank 32; shade = drop relative
to full rank; the default r = 32 is outlined.

**New-baselines figure.** FLAME-CL against the MoE-based (TRGE, SAME) and distillation (LwF) baselines on Setups 1–3 (seed 42; Setup 4
in Tabs. 13–14). Parameter panels are inference-time totals: TRGE, SAME and LwF adapt only the MoE and keep the full encoder.

**Spectra/alignment figure.** Full captions for both paths are in `tables/cl_alignment_snippets.tex`.

## 2. Which runs are used (single source of truth: `scripts/cl_data.py`)

All runs are under `src/checkpoints/continual/laplace/<sequence>/<modalities>/<seed>/`.

- **FLAME-CL**: `cl_ours_target_moe_and_encoder_…_wd1.0`, with encoder policy `first_appearance`.
  - This is the configuration of Sec. 2.2 / App. E, and it is the run behind Tab. 3 (AA 0.793, F 0.000; −cursor 0.759/0.055).
  - Setup 2 only has a `wd0.1` run of it.
  - The old Fig. 5 registry (`src/aggregate_continual.py`) mixed in the `enc_all_trainable` variant. That variant is not strictly
    forgetting-free: in Setup 2, MOR rises 0.826 → 0.829 after stage 2.
- **LoRA**: `cl_lora_lorarank32_router_per_task_router_fixed_experts_rank32`, i.e. rank-32 adapters on the **MoE experts only**.
  - This is the baseline the original paper figures plotted (old Fig. 9 and the registry), and the one the authors intend.
  - App. D.3.1's sentence "LoRA grows from 109.54M to 125.29M encoder parameters" quotes a different run (encoder + MoE adapters).
- **LoRA (encoder+MoE)**: `cl_lora_lorarank32_target_moe_and_encoder_…`, i.e. rank-32 adapters on the encoders and the MoE.
  - Shown only as an extra variant, in its own figure and tables. Fig. 5 keeps the MoE-only baseline.
  - It is the run behind the D.3.1 sentence: 109.54 → 125.29M method-attributable encoder parameters in Setup 4.
  - Only seed 42 exists for either LoRA variant.
- **Simple FT / EWC**: `cl_ewc_lamb{0,1}…_enc_all_trainable_target_moe_and_encoder`.
- **LwF / SAME (no-mask) / TRGE / Lifelong-PT**: the same directories as the registry.
- **Seeds**: seed 42 everywhere, except the Setup 4 multi-seed results (703, 8622, 92784); those are the only seeds on disk for those methods.
- **Parameter counts**: figures default to **inference-time totals** (`param_count_progression.txt`), as the original Fig. 5/9 did.
  - Method-attributable counts cannot compare methods that adapt different parts: LoRA, LwF, SAME, TRGE and Lifelong-PT adapt only the
    MoE, so their attributable encoder count is 0, although they keep the full 153M encoder.
- **Checks**: the loader reproduces Tab. 14 (every F, BWT and AA cell) and every row of Tab. 3 (AA, F, params) exactly.
- **Spectra/alignment** (same FLAME-CL runs, seed 42; Setup 4 also seeds 703/8622/92784):
  - **Path A source:** `src/analysis/eval_cl_stage_geometry.py`. It takes the untruncated stage component and the routed-input second
    moments on the first ≤512 validation samples of each task.
  - **Path B / Fig. 16 source:** `eval_cl_spectrum.py` on test batches. Fig. 16 used 10 batches; Setup 2 uses 64, because eICU gives
    5 tokens per sample and 10 batches leave about 160 tokens per expert for a 128×128 covariance.
  - **Agreement:** the two agree within ±1–2 ranks on Setups 1 and 3. Setup 4 stage 1 differs (input r90 4 vs 8) because its 10 test
    batches are dominated by EMBED (512 samples per batch vs 8 for MIMIC).
  - **Checks:** the weight spectra are identical between the two. The kept rank-32 factors match the top-32 SVD of the untruncated
    component (relative difference 9e-7).

## 3. Text the new numbers contradict (please check before submitting)

1. **Lifelong-PT vs. LoRA (the requested fix).** Figs. 5/10 now show Simple FT, EWC, LoRA, FLAME-CL. This matches the intro, Sec. 3.2,
   RQ3, RQ4, App. D.3.1 and the Fig. 8 caption. Lifelong-PT stays in Fig. 8.
2. **"5–15× fewer encoder parameters than fine-tuning, EWC and LoRA"** (intro, Sec. 3.2, RQ4, conclusion, App. G).
   - **Inference totals (what Fig. 5 shows):** FLAME-CL stores 2.3–3.0× fewer encoder parameters than all three in Setups 1 and 4,
     1.2–3.7× in Setup 3 and 1.0–1.1× in Setup 2.
   - **Method-attributable counts:** "5–15×" exists only against Simple FT/EWC in Setups 1 and 4 (4.6–14×). It cannot include LoRA,
     which adds no encoder parameters.
   - Suggested: "2.3–3× fewer parameters at inference than Simple FT, EWC and LoRA on the sequences with MIMIC-IV encoders".
   - **Against encoder+MoE LoRA:** 5.3× and 5.2× fewer method-attributable encoder parameters in Setups 1 and 4 (1.8–1.9× in
     Setups 2–3), and 2.5× fewer at inference. So "5–15× fewer than fine-tuning, EWC and LoRA" holds on method-attributable counts
     in Setups 1 and 4, provided "LoRA" means the encoder+MoE variant there.
3. **App. D.3.1 parameter paragraph.** In Setup 4, at inference:
   - Simple FT, EWC and LoRA all sit flat at 153.0M encoder parameters.
   - FLAME-CL grows 51.4 → 67.4M.
   - MoE: FLAME-CL 0.19 → 0.56M vs. LoRA 0.43 → 0.80M. The current attributable MoE numbers, 0.17 → 0.50 vs. 0.41 → 0.74, are also
     correct for MoE-only LoRA.
   - The "LoRA grows from 109.54M to 125.29M" sentence is correct for **encoder+MoE** LoRA (method-attributable). Keep it only if
     the text names that variant (e.g. next to `tab_lora_variants_summary.tex`); otherwise delete it.
4. **LoRA comparison claims** (D.3.1, RQ3, D.3.8, Fig. 9 caption). The old text matches the **encoder+MoE** run, while the old
   figures plotted the **MoE-only** run. Now that both are reported, say which LoRA each claim is about:
   - **vs encoder+MoE LoRA:** "within 0.01 AA on every sequence" is true (−0.008 to +0.002). Neither variant forgets. FLAME-CL needs
     2.5× fewer encoder parameters at inference in Setups 1 and 4.
   - **vs MoE-only LoRA:** the points below.
   - **AA:** "FLAME-CL is within 0.01 of LoRA on every sequence" becomes "FLAME-CL exceeds LoRA by 0.015–0.029 AA on every sequence".
   - **Retention:** "LoRA matches retention only by freezing the base" is false in Setup 4. LoRA keeps old tasks in Setups 1–3 but loses
     DENSITY 0.92 → 0.82 and PHENO 0.71 → 0.69 there (F 0.026).
   - **Plasticity:** LoRA is also less plastic: RAD 0.68 vs 0.74, BIRADS 0.74 vs 0.81, LOS 0.79 vs 0.82.
   - **Fig. 9 caption:** "tracks LoRA's within 0.01 at every stage" and "encoder parameters grow linearly in both" are false, because
     LoRA's encoder is flat. Since Fig. 5 now contains LoRA, dropping Fig. 9 and D.3.8 is simplest.
5. **"FLAME-CL retains end-of-stage AUROC within 0.01 of each task's first-introduction value across four sequences"** (Sec. 3.2, RQ3, D.8).
   This fails for Setup 3 DENSITY: 0.926 → 0.902 (−0.024), the one-time rank-32 truncation cost.
6. **"Simple FT drops 0.03–0.05 on the earliest tasks"** (RQ3, D.8) and **"beats Simple FT and EWC by 0.02–0.05"** (D.3.1). These are true
   for PHENO and MOR, but DENSITY drops 0.16 (Setups 3, 4), where FLAME-CL beats them by 0.11–0.16.
7. **Tab. 13 FLAME-CL row** (also rebuttal Table A and Tab. 7). Regenerated from the runs that reproduce Tabs. 3/14:
   RAD 0.75 → **0.73**, RISK 0.75 → **0.74**, MOR first 0.83 → 0.82, DENSITY first 0.92 → 0.91, PHENO final 0.70 → 0.71.
   - **Knock-on (D.3.2):** "reaching 0.75 on RAD … EWC matches it" becomes "0.73; EWC exceeds it (0.75)".
   - **Knock-on (D.1):** "matches or exceeds the jointly trained model on five of eight tasks" becomes **four**.
   - **Baseline cells:** several are off by about 0.01 (TRGE BIRADS/MOR/RISK; SAME DENSITY/LOS/BIRADS/MOR/IHM/RISK); the full table is
     in `tables/tab13_tab14_setup4.tex`.
8. **Tab. 15 / D.3.3 (Setup 2):** FLAME-CL RAD is **0.74**, not 0.75; the 0.75 came from the `enc_all_trainable` run. The conclusion still
   holds: every baseline that retains MOR reaches at most 0.69.
9. **App. D.3.7 / Fig. 8 caption: "FLAME-CL matches Lifelong-PT on AUROC at every stage"** is wrong in FLAME-CL's favour.
   - **Setup 4 forgetting:** Lifelong-PT forgets PHENO 0.71 → 0.57 and DENSITY 0.92 → 0.51 (AA 0.70 vs 0.79; F 0.107 vs 0).
   - **New tasks:** Lifelong-PT trails on every new task.
   - **Colour:** Lifelong-PT is now violet, not blue.
10. **D.2.5:** DIAG varies by **0.036**, not 0.037. 48-IHM's 0.014 is confirmed as the largest other range.
11. **D.6:** fill in from `tables/robustness_spearman.md`.
    - **Routing reaction vs. importance:** ρ = +0.14, p = 0.44 (n = 31).
    - **Fusion share vs. importance:** ρ = +0.73, p = 2.7×10⁻⁶; ρ = +1 on 5 of 8 tasks, but only MOR is significant on its own.
    - **Monotonicity:** JSD rises monotonically with noise for all 31 pairs.
    - **Spillover:** exactly 0.
    - **Tab. 17 medians:** −4.8% / −15.1%, reproduced.
12. **If you keep Tab. 3**, its "–" parameter cells can be filled with inference totals: TRGE 160.9M, SAME 160.0M, LwF 160.1M, EWC 160.0M.
13. **"Zero-initialized additive component" (Sec. 2.2, Alg. 1, App. B.2) and App. D.4.3.**
    - **What the code does:** it zero-initialises only the new component's output layer fc2 (`src/continual/cl_moe.py: zero_init_fc2`).
      fc1 and the temporal conv keep PyTorch's default random init. The component therefore starts as an exact no-op, but not every
      matrix starts at zero.
    - **What the new figure shows:** Proposition 2's prediction holds for the zero-initialised fc2.
      - Data-aware r90 is 3 (range 1–8).
      - At least 99.8% of the data-aware energy is kept at r = 32.
      - Alignment A(16) = 0.41 against 0.125 random, on all four sequences.
    - **The random-init fc1** stays near chance alignment (A(16) = 0.16). Its data-aware energy is not low-rank (r90 ≈ 66; 65% kept at
      r = 32). Its truncation is safe only behaviourally (Tab. 4 rank sweep; the first → final AUROC of Tab. 13 and Fig. 5).
    - **Current D.4.3 text:** it says "one group of sublayers concentrates nearly all of its energy in the first few ranks"; those are
      the fc2s.
    - **Suggested wording** for Sec. 2.2 and a replacement D.4.3 paragraph are in `tables/cl_alignment_snippets.tex`.
    - **Stage numbering:** Fig. 16 (its PNG titles and subcaptions) numbers stages from 0, while Fig. 5 and Tabs. 13/14 number them from
      1. The new figure and table use 1–3.

Also: `src/results/flame_aux_ablation/aux_ablation_tables.md` (old) has Mortality and Readmission swapped. Its `.tex`/`.csv` siblings and the
new tables here are correct.

## 4. Layout and length (60 pages)

**Page 9 (too many tables).** Replace Tabs. 3–4 with `fig_cl_summary_tab3_tab4.pdf`. It is a little taller (2.25 in) but reads at a glance,
and the exact values stay in Tabs. 14 and 16. If you keep the tables instead, do not move them to page 10: page 10 is full with Fig. 5,
Tab. 5 and the conclusion.

**Fig. 5 height.** Use `fig5_cl_trajectories_compact.pdf` (4.4 in, about the old figure's height) so the main text does not grow.
The taller variant is about 1.8 in more.

**Page 35.** The old Fig. 10 plus four lines of text, then the full-page spectra figures. The new Fig. 10 is one compact figure, but the gap
comes from the floats that follow. Let the D.5 text flow there (remove a `\clearpage`/`\FloatBarrier`, or give Figs. 11–16 `[p]`), or
condense Figs. 11–14 (below).

**Page 49.** Tab. 19 alone, because Fig. 27 (five full-width rows) is a full-page float. Either show stages 0/2/4 only or shrink the rows
so Tab. 19 and Fig. 27 share a page, or let the D.8/E text follow Tab. 19.

**Spectra/alignment (App. D.4.3).** Path A (one 4.7 in figure + table, about one page) takes the place of today's full-page Fig. 16, so
covering all four sequences costs about nothing. Without the table (numbers are in the D.4.3 paragraph) it saves about 0.4 page. Path B
(Fig. 16 format for Setups 1–3) adds about 3 pages.

**Trim candidates** (about 9 pages in total):

| Cut | Saves | Why it is safe |
|---|---|---|
| App. D.8, "Extended Discussion of RQs" | ~1.5 pp | restates Sec. 3.3; several of its numbers are now out of date (items 4–6 above) |
| App. D.3.8 + Fig. 9 | ~0.8 pp | LoRA is now in Fig. 5 |
| Figs. 11–14 (single-task spectra, 4 pages) → one compact grid, or point to the code README | ~3 pp | the joint (Fig. 15) and continual (Fig. 16) spectra carry the argument |
| Single-task routing (Figs. 17–20) and per-setup continual routing (Figs. 23–25) | ~3 pp | keep Fig. 22 (all tasks) and Fig. 26 (Setup 4); the rest can live in the README |
| Tabs. 9–10 or Fig. 7 (number of experts) | ~0.5 pp | same numbers twice |
| Fig. 8 folded into the new-baselines figure (Lifelong-PT as a 4th baseline) | ~0.8 pp | one figure instead of two |

The NeurIPS checklist (pp. 53–60) is mandatory; only your justification text can be shortened.

## 5. Remaining to-dos (in the paper source, which is not on io79)

1. Insert the files in §1 with the suggested captions; pick compact vs. tall Fig. 5, and tables vs. figure for Tabs. 3–4.
2. Make the text edits in §3 (items 2–9 change numbers or claims; 1, 10–12 are small).
3. Spectra + alignment for all four sequences (rebuttal promise to 9k3u): pick path A or B (§1, last row; §4). Update D.4.3 with the
   paragraph in `tables/cl_alignment_snippets.tex`, and the "zero-initialized" wording in Sec. 2.2 (§3 item 13).
4. Apply the layout and trim choices in §4, recompile, and check that the main text is still within the page limit and that all
   cross-references resolve.

## 6. Scripts

| Script | Output |
|---|---|
| `scripts/cl_data.py` | run registry + loaders (imports the parsers from `src/aggregate_continual.py`) |
| `scripts/make_cl_figures.py` | Figs. 5, 8, 9, 10, the new-baseline figures and the LoRA-variants figure |
| `scripts/make_cl_summary_figure.py` | figure version of Tabs. 3–4 |
| `scripts/make_cl_tables.py` | CL tables + `data/cl_trajectories_all.csv` |
| `scripts/make_aux_loss_tables.py` | per-task loss-weight tables (reads raw per-seed results via `src/make_aux_ablation_tables.py`) |
| `scripts/make_robustness_stats.py` | D.6 statistics from the sweep JSON |
| `scripts/make_cl_alignment_figure.py` | continual-phase spectra + alignment figures/table; copies the Fig. 16-format panels. Inputs come from `src/analysis/run_cl_stage_geometry.sh` and `src/analysis/run_cl_spectrum.sh` (both now take `TASK_RAW`/`MODELTYPE`/`WD` for Setups 1–3; usage in their headers) |
| `scripts/paper_style.py` | shared matplotlib style; colours checked with the dataviz CVD validator (FLAME-CL red, LoRA green, Simple FT blue, EWC orange) |
| `make_all.sh` | runs everything and builds `preview/preview.pdf` |
