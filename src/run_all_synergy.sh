#!/usr/bin/env bash

# Batch runner for task-pair synergy using src/synergy.py.
# It assumes you have 2-task FuseMoE checkpoints saved under:
#   src/checkpoints/fusemoe/multitask/<taskA>-<taskB>/
# with encoder filenames following the pattern:
#   mimic_iv_<taskA>-<taskB>_TS-Text-CXR_cc-mlo-2dcc-2dmlo_<TASK>_encoder.pt
#
# The output is a CSV at src/results/task_synergy.csv with columns:
#   task_a,task_b,synergy

set -euo pipefail

export CUDA_VISIBLE_DEVICES=3

PROJECT_ROOT="/cis/home/xhan56/code/clinical-highmmt"
SRC_DIR="$PROJECT_ROOT/src"
CHECKPOINT_DIR="$SRC_DIR/checkpoints/fusemoe/multitask"
RESULTS_DIR="$SRC_DIR/results"
OUT_CSV="$RESULTS_DIR/task_synergy.csv"

mkdir -p "$RESULTS_DIR"

echo "Writing synergy results to $OUT_CSV"
echo "task_a,task_b,synergy" > "$OUT_CSV"

run_pair () {
  local pair="$1"    # e.g. ihm-risk
  local task_a="$2"  # e.g. ihm
  local task_b="$3"  # e.g. risk

  local enc_dir="$CHECKPOINT_DIR/$pair"

  # Map logical task names to suffixes used in encoder filenames.
  # For MIMIC tasks, suffix is IHM/LOS; for EMBED tasks, BIRADS/RISK/DENSITY;
  # for eICU tasks, RAD (readmission) and MOR (mortality).
  local suffix_a
  case "$task_a" in
    readmission) suffix_a="RAD" ;;
    mortality)   suffix_a="MOR" ;;
    *)           suffix_a="${task_a^^}" ;;
  esac

  local suffix_b
  case "$task_b" in
    readmission) suffix_b="RAD" ;;
    mortality)   suffix_b="MOR" ;;
    *)           suffix_b="${task_b^^}" ;;
  esac

  # Encoder filenames may contain different modality strings
  # (e.g. TS-Text-CXR_cc-mlo-2dcc-2dmlo vs T1-T2-T3-T4-T5_cc-mlo-2dcc-2dmlo),
  # so we wildcard that part and only fix the prefix/suffix.
  local enc_a
  enc_a=$(ls "$enc_dir"/mimic_iv_${pair}_*_${suffix_a}_encoder.pt 2>/dev/null | head -n 1 || true)
  local enc_b
  enc_b=$(ls "$enc_dir"/mimic_iv_${pair}_*_${suffix_b}_encoder.pt 2>/dev/null | head -n 1 || true)

  if [[ ! -f "$enc_a" ]]; then
    echo "[WARN] Missing encoder A for $pair: $enc_a"
    return
  fi
  if [[ ! -f "$enc_b" ]]; then
    echo "[WARN] Missing encoder B for $pair: $enc_b"
    return
  fi

  echo "=== Running synergy for $pair ($task_a vs $task_b) ==="

  # Run synergy.py and capture the line containing "Synergy:"
  local sy_line
  sy_line=$(cd "$SRC_DIR" && python -W ignore synergy.py \
      --encoder_a_path "$enc_a" \
      --encoder_b_path "$enc_b" \
      --task "$pair" \
      --mimic_path '/export/io79/data/schaud35/datasets/' \
      --eicu_path '/export/io79/data/schaud35/datasets/eicu/processed/' \
      --embed_path '/export/io79/data/schaud35/datasets/EMBED' \
      --train_bs_mimic 8 --train_bs_eicu 128 --train_bs_embed 512 --eval_batch_size 8 \
      --ihm_mod 'TS-Text-CXR' \
      --los_mod 'TS-Text-CXR' \
      --rad_mod 'T1-T2-T3-T4-T5' \
      --mor_mod 'T1-T2-T3-T4-T5' \
      --birads_mod 'cc-mlo-2dcc-2dmlo' \
      --risk_mod 'cc-mlo-2dcc-2dmlo' \
      --density_mod 'cc-mlo-2dcc-2dmlo' \
      --tt_max 48 --tt_max_eicu 1 --embed_time 64 \
      --num_heads 8 --layers 3 --kernel_size 1 \
      --model_name "bioLongformer" --fusion_model 'fusemoe' \
      --shared_modality_encoders --use_pt_text_embeddings --multitask_moe \
      --reg_ts --TS_mixup --fp16 \
      --num_train_epochs 10 \
    | grep "Synergy:" || true)

  if [[ -z "$sy_line" ]]; then
    echo "[WARN] No synergy line found for $pair; skipping."
    return
  fi

  # Extract numeric value from the line containing "Synergy:"
  # Example line: "Optimization Complete. Synergy: 3.731377"
  local val
  val=$(echo "$sy_line" | awk '{print $NF}')

  echo "$task_a,$task_b,$val" >> "$OUT_CSV"
}

###############################################################################
# Task pairs to evaluate
#
# Add/remove pairs here depending on which 2-task FuseMoE checkpoints you have.
###############################################################################

# MIMIC vs EMBED
run_pair "ihm-risk"      "ihm"        "risk"
run_pair "ihm-birads"    "ihm"        "birads"
run_pair "ihm-density"   "ihm"        "density"
run_pair "los-risk"      "los"        "risk"
run_pair "los-birads"    "los"        "birads"
run_pair "los-density"   "los"        "density"

# eICU vs EMBED
run_pair "readmission-risk"   "readmission" "risk"
run_pair "readmission-birads" "readmission" "birads"
run_pair "readmission-density" "readmission" "density"
run_pair "mortality-risk"     "mortality"   "risk"
run_pair "mortality-birads"   "mortality"   "birads"
run_pair "mortality-density"  "mortality"   "density"

# MIMIC vs eICU (if these checkpoints exist)
run_pair "ihm-readmission"    "ihm"        "readmission"
run_pair "ihm-mortality"      "ihm"        "mortality"
run_pair "los-readmission"    "los"        "readmission"
run_pair "los-mortality"      "los"        "mortality"

echo "Done. Results in $OUT_CSV"

