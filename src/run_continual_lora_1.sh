export CUDA_VISIBLE_DEVICES=2

# Continual learning launcher (parallel to run_flame_embed.sh).
# Step-1: naive sequential per-task training. The CL-specific args
# (--reserved_rank, --router_growth_mode, --replay_proportion) are accepted
# but only consumed in Step-2+ (rank reservation + expert pool growth).
SEEDS=(42)
CL_METHODS=('lora')
FUSION_MODEL='fusemoe'
GATING='laplace'
MOD_DROP_RATE='0.0'
EXPERTS=5
# TASK_RAW='ihm-mortality-birads;pheno-readmission-density;los-diag-risk'
# TASK_RAW='pheno-mortality;los-readmission;ihm-birads-density''
TASK_RAW='pheno-density;los-birads-mortality;ihm-risk-readmission' # 'pheno-birads;los-risk;ihm-density' # 'pheno-birads;los-mortality-risk;ihm-readmission-density'
# Path-safe form of TASK_RAW: ';' -> '__' (matches _path_safe_task_str in continual_tasks.py).
TASK_PATH="${TASK_RAW//;/__}"
EWC_LAMB=1
EWC_ALPHA=0.5
EWC_FI_SAMPLING='true'
EWC_FI_NUM_SAMPLES=-1
# --router_expansion (default): per-stage ModalityRouter heads (current behavior).
# --no-router_expansion (only meaningful for --cl_method=ewc): single shared
#   router across all stages; EWC also regularizes its w_gate / w_noise.
ROUTER_EXPANSION_FLAG='--no-router_expansion' #'--router_expansion'

# Heads-only ablation: freeze the entire MoE/cross-attn backbone at random
# init for every stage; only per-task heads + per-task projections train.
# Encoders follow the standard first-appearance unfreeze rule. Set to
# '--heads_only' to enable; leave empty to use the standard backbone training.
HEADS_ONLY_FLAG='' #'--heads_only' or ''

# Encoder freeze policy at stage > 0 (and at every stage under --heads_only):
#   first_appearance (default): unfreeze only encoders whose modality first
#                               appears at the current stage.
#   all_frozen:                 keep all encoders frozen at stage > 0.
#   all_trainable:              unfreeze every encoder param at every stage.
ENCODER_FREEZE_MODE='all_trainable' # 'all_trainable' #'first_appearance'

# SVD-reservation scope (only meaningful for --cl_method=ours):
#   moe (default):       SVD-truncate / stack only MoE expert weights.
#   moe_and_encoder:     also SVD-truncate / stack every trainable encoder
#                        nn.Linear and nn.Conv1d(k=1). Each gets a
#                        StackedLowRank* wrapper so subsequent stages add
#                        a fresh trainable low-rank correction on top.
#                        Overrides --encoder_freeze_mode for wrapped layers.
CL_TARGET='moe_and_encoder' # 'moe' or 'moe_and_encoder'

RESULTS_DIR='/cis/home/schaud35/clinical-highmmt/src/results'

for SEED in "${SEEDS[@]}"; do
    for CL_METHOD in "${CL_METHODS[@]}"; do
        python -W ignore continual_tasks.py --num_train_epochs 5 \
                        --kernel_size 1 --train_bs_mimic 8 --train_bs_eicu 128 --train_bs_embed 512 \
                        --eval_batch_size 8 --seed "${SEED}" \
                        --gradient_accumulation_steps 16  --num_update_bert_epochs 2 --bertcount 0 \
                        --ts_learning_rate 0.0004 --txt_learning_rate 0.00002 \
                        --notes_order 'Last' --num_of_notes 5 --max_length 1024 --layers 1 --cross_layers 1\
                        --embed_dim 128 \
                        --perceiver_dim 64 \
                        --model_name "bioLongformer"\
                        --task "${TASK_RAW}" \
                        --ihm_mod 'TS-Text-CXR'\
                        --los_mod 'TS-Text-CXR'\
                        --pheno_mod 'TS-Text-CXR'\
                        --rad_mod 'T1-T2-T3-T4-T5'\
                        --mor_mod 'T1-T2-T3-T4-T5'\
                        --birads_mod 'cc-mlo-2dcc-2dmlo'\
                        --risk_mod 'cc-mlo-2dcc-2dmlo'\
                        --density_mod 'cc-mlo-2dcc-2dmlo'\
                        --diag_mod 'I-G-C-B' \
                        --mimic_path '/export/io79/data/schaud35/datasets/'\
                        --eicu_path '/export/io79/data/schaud35/datasets/eicu/processed/'\
                        --adni_path '/export/io79/data/schaud35/datasets/adni/adni_processed/'\
                        --embed_path '/export/io79/data/schaud35/datasets/EMBED'\
                        --num_heads 8\
                        --embed_time 64\
                        --tt_max 48\
                        --tt_max_eicu 1\
                        --TS_mixup \
                        --mixup_level 'batch'\
                        --cross_method 'moe'\
                        --fp16 \
                        --reg_ts \
                        --balance_loss_coef 1.0 \
                        --fusion_model "${FUSION_MODEL}" \
                        --num_of_experts "${EXPERTS}" \
                        --top_k 2 4 \
                        --router_type 'joint' \
                        --gating_function "${GATING}" \
                        --use_pt_text_embeddings \
                        --shared_modality_encoders \
                        --modality_drop_rate "${MOD_DROP_RATE}" \
                        --multitask_moe \
                        --alpha 'const_0.0' \
                        --lr 0.0001 \
                        --weight_decay 0.1 \
                        --reserved_rank 32 \
                        --router_growth_mode 'per_task_router' \
                        --router_combine 'mean' \
                        --replay_proportion 0.0 \
                        --fixed_experts \
                        --cl_method "${CL_METHOD}" \
                        --lora_cl_rank 32 \
                        --ewc_lamb "${EWC_LAMB}" \
                        --ewc_alpha "${EWC_ALPHA}" \
                        --ewc_fi_sampling "${EWC_FI_SAMPLING}" \
                        --ewc_fi_num_samples "${EWC_FI_NUM_SAMPLES}" \
                        "${ROUTER_EXPANSION_FLAG}" \
                        --encoder_freeze_mode "${ENCODER_FREEZE_MODE}" \
                        --cl_target "${CL_TARGET}" \
                        ${HEADS_ONLY_FLAG}
    done
done

# Aggregate per CL method, per stage. The CL writer puts each stage's
# best_model_results files under .../experts_E/stage{s}_<label>/<seed>/...
# so aggregate_results.py can be pointed at each stage subtree directly.
# task_names per stage matches what the after-stage eval covers (all tasks
# in stages 0..s, since the full-rank eval iterates the cumulative set).
for CL_METHOD in "${CL_METHODS[@]}"; do
    BASE_DIR="${RESULTS_DIR}/${FUSION_MODEL}/continual_${CL_METHOD}/${GATING}/${TASK_PATH}/mod_drop_rate_${MOD_DROP_RATE}/experts_${EXPERTS}/seed_${SEED}"
    if [ ! -d "${BASE_DIR}" ]; then
        echo "[aggregate] no results for ${CL_METHOD} under ${BASE_DIR}; skipping."
        continue
    fi
    # Stage 0 covers ['ihm', 'los']; stage 1 also covers BIRADS.
    for STAGE_DIR in "${BASE_DIR}"/stage*_*; do
        [ -d "${STAGE_DIR}" ] || continue
        STAGE_NAME=$(basename "${STAGE_DIR}")
        case "${STAGE_NAME}" in
            stage0_*) TASK_NAMES='ihm,los' ;;
            stage1_*) TASK_NAMES='ihm,los,birads' ;;
            *)        TASK_NAMES='' ;;
        esac
        echo "[aggregate] ${CL_METHOD} ${STAGE_NAME} with task_names=${TASK_NAMES}"
        if [ -n "${TASK_NAMES}" ]; then
            python aggregate_results.py --result_dir "${STAGE_DIR}" --task_names "${TASK_NAMES}"
        else
            python aggregate_results.py --result_dir "${STAGE_DIR}"
        fi
    done
done

# --use_wandb \
                    # --wandb_project 'clinical-highmmt'
                    # --cl_method options:
                    #   'ours'  -> stacked low-rank components (current pipeline)
                    #   'lora'  -> stage-0 full-rank base + per-stage LoRA adapters
                    #              on temporal_conv/fc1/fc2 weights only.
                    #              Requires --fixed_experts and per_task_router.