#!/bin/bash
# Rebuild every figure, table and the preview PDF in presentations/paper_updates.
set -euo pipefail
cd "$(dirname "$0")/scripts"
PY="${PY:-$HOME/anaconda3/envs/mmt/bin/python}"
$PY make_cl_figures.py
$PY make_cl_summary_figure.py > /dev/null
$PY make_cl_tables.py
$PY make_aux_loss_tables.py > /dev/null
$PY make_robustness_stats.py > /dev/null
$PY make_cl_alignment_figure.py > /dev/null
cd ../preview
pdflatex -interaction=nonstopmode -halt-on-error preview.tex > build.log
pdflatex -interaction=nonstopmode -halt-on-error preview.tex > build.log
echo "done: $(pwd)/preview.pdf"
