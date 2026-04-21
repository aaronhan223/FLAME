

# Select the same GPU used for your training runs
export CUDA_VISIBLE_DEVICES=3

# Define the root directory of your project
PROJECT_ROOT="/cis/home/xhan56/code/clinical-highmmt"
CHECKPOINT_DIR="$PROJECT_ROOT/src/checkpoints/fusemoe/multitask"

# Path to the M4oE encoders

# ihm-risk
# ENC_A="$CHECKPOINT_DIR/ihm-risk/mimic_iv_ihm-risk_TS-Text-CXR_cc-mlo-2dcc-2dmlo_IHM_encoder.pt"
# ENC_B="$CHECKPOINT_DIR/ihm-risk/mimic_iv_ihm-risk_TS-Text-CXR_cc-mlo-2dcc-2dmlo_RISK_encoder.pt"

# ihm-birads
ENC_A="$CHECKPOINT_DIR/ihm-birads/mimic_iv_ihm-birads_TS-Text-CXR_cc-mlo-2dcc-2dmlo_IHM_encoder.pt"
ENC_B="$CHECKPOINT_DIR/ihm-birads/mimic_iv_ihm-birads_TS-Text-CXR_cc-mlo-2dcc-2dmlo_BIRADS_encoder.pt"

python -W ignore synergy.py \
    --encoder_a_path "$ENC_A" \
    --encoder_b_path "$ENC_B" \
    --task 'ihm-birads' \
    --hidden_size 128 \
    --embed_dim 128 \
    --perceiver_dim 64 \
    --num_of_experts 3 5 \
    --top_k 2 4 \
    --gating_function "laplace" \
    --multitask_moe \
    --mimic_path '/export/io79/data/schaud35/datasets/' \
    --eicu_path '/export/io79/data/schaud35/datasets/eicu/processed/' \
    --embed_path '/export/io79/data/schaud35/datasets/EMBED' \
    --train_bs_mimic 8 --train_bs_eicu 128 --train_bs_embed 512 --eval_batch_size 8 \
    --ihm_mod 'TS-Text-CXR' \
    --rad_mod 'T1-T2-T3-T4-T5' \
    --density_mod 'cc-mlo-2dcc-2dmlo' \
    --risk_mod 'cc-mlo-2dcc-2dmlo' \
    --birads_mod 'cc-mlo-2dcc-2dmlo' \
    --tt_max 48 --tt_max_eicu 1 --embed_time 64 \
    --num_heads 8 --layers 3 --kernel_size 1 \
    --model_name "bioLongformer" --fusion_model 'fusemoe' \
    --shared_modality_encoders --use_pt_text_embeddings --multitask_moe \
    --reg_ts --TS_mixup --fp16 \
    --num_train_epochs 10