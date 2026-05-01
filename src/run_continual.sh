export CUDA_VISIBLE_DEVICES=2

# Continual learning launcher (parallel to run_flame_embed.sh).
# Step-1: naive sequential per-task training. The CL-specific args
# (--reserved_rank, --router_growth_mode, --replay_proportion) are accepted
# but only consumed in Step-2+ (rank reservation + expert pool growth).

python -W ignore continual_tasks.py --num_train_epochs 5 \
                --kernel_size 1 --train_bs_mimic 8 --train_bs_eicu 128 --train_bs_embed 512 \
                --eval_batch_size 8 --seed 42 \
                --gradient_accumulation_steps 16  --num_update_bert_epochs 2 --bertcount 0 \
                --ts_learning_rate 0.0004 --txt_learning_rate 0.00002 \
                --notes_order 'Last' --num_of_notes 5 --max_length 1024 --layers 1 --cross_layers 1\
                --embed_dim 128 \
                --perceiver_dim 64 \
                --model_name "bioLongformer"\
                --task 'ihm-los;birads'\
                --ihm_mod 'TS-Text'\
                --los_mod 'TS-CXR'\
                --pheno_mod 'TS-Text-CXR'\
                --rad_mod 'T1-T2-T3-T4-T5'\
                --mor_mod 'T1-T2-T3-T4-T5'\
                --birads_mod 'cc-mlo-2dcc-2dmlo'\
                --risk_mod 'cc-mlo-2dcc-2dmlo'\
                --density_mod 'cc-mlo-2dcc-2dmlo'\
                --mimic_path '/export/io79/data/schaud35/datasets/'\
                --eicu_path '/export/io79/data/schaud35/datasets/eicu/processed/'\
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
                --fusion_model 'fusemoe' \
                --num_of_experts 5 \
                --top_k 2 4 \
                --router_type 'joint' \
                --gating_function "laplace" \
                --use_pt_text_embeddings \
                --shared_modality_encoders \
                --modality_drop_rate 0.0 \
                --multitask_moe \
                --alpha 'const_0.0' \
                --lr 0.0001 \
                --weight_decay 0.1 \
                --reserved_rank 16 \
                --router_growth_mode 'per_task_router' \
                --router_combine 'mean' \
                --replay_proportion 0.0 \
                --fixed_experts \
                # --use_wandb \
                # --wandb_project 'clinical-highmmt'
