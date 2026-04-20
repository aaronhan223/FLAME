#!/bin/bash
cd /cis/home/schaud35/clinical-highmmt
export CUDA_VISIBLE_DEVICES=6

# ============================================================
# FlexMoE IHM — low-rank expert evaluation
# ============================================================
echo "===== Evaluating FlexMoE model with low-rank experts ====="
python -W ignore -m src.analysis.eval_lowrank_experts_flexmoe \
    --model_path /cis/home/schaud35/clinical-highmmt/src/checkpoints/flexmoe/multitask/ihm/ihm_TS-Text_mod_drop_rate_0.0.pt \
    --encoder_path \
        /cis/home/schaud35/clinical-highmmt/src/checkpoints/flexmoe/multitask/ihm/ihm_TS-Text_mod_drop_rate_0.0_IHM_mod_drop_rate_0.0_encoder.pt \
    --ranks 0 1 2 4 8 full \
    --output_dir /cis/home/schaud35/clinical-highmmt/src/analysis/analysis_results/lowrank_eval/flexmoe-ihm \
    --task ihm \
    --ihm_mod 'TS-Text' \
    --los_mod 'TS-CXR' \
    --pheno_mod 'TS-Text-CXR' \
    --rad_mod 'T1-T2-T3-T4-T5' \
    --mor_mod 'T1-T2-T3-T4-T5' \
    --birads_mod 'cc-mlo-2dcc-2dmlo' \
    --risk_mod 'cc-mlo-2dcc-2dmlo' \
    --density_mod 'cc-mlo-2dcc-2dmlo' \
    --mimic_path '/export/io79/data/schaud35/datasets/' \
    --eicu_path '/export/io79/data/schaud35/datasets/eicu/processed/' \
    --embed_path '/export/io79/data/schaud35/datasets/EMBED' \
    --kernel_size 1 --train_bs_mimic 8 --eval_batch_size 8 --seed 42 \
    --gradient_accumulation_steps 16 --num_update_bert_epochs 2 --bertcount 0 \
    --ts_learning_rate 0.0004 --txt_learning_rate 0.00002 \
    --notes_order 'Last' --num_of_notes 5 --max_length 1024 --layers 3 \
    --embed_dim 128 --perceiver_dim 64 \
    --model_name "bioLongformer" \
    --num_heads 8 --embed_time 64 --tt_max 48 --tt_max_eicu 1 \
    --TS_mixup --mixup_level 'batch' \
    --cross_method 'moe' --fp16 --reg_ts \
    --balance_loss_coef 0.01 \
    --fusion_model 'flexmoe' \
    --num_of_experts 3 5 --top_k 2 4 \
    --router_type 'joint' --gating_function "laplace" \
    --use_pt_text_embeddings --shared_modality_encoders \
    --modality_drop_rate 0.0 \
    --num_train_epochs 0
