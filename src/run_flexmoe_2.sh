export CUDA_VISIBLE_DEVICES=7
SEEDS=(0 42 453) #(0 42 453 1002 10293)
EXPERTS=(5)

RESULTS_DIR='/cis/home/schaud35/clinical-highmmt/src/results'
FUSION_MODEL='flexmoe'
TASKS=('ihm' 'los' 'pheno')
GATING='laplace'
BALANCE_LOSS_COEF='1.0'
ALPHA='const_0.0'
MOD_DROP_RATE='0.0'

for TASK in "${TASKS[@]}"; do
    for EXPERT in "${EXPERTS[@]}"; do 
        # for SEED in "${SEEDS[@]}"; do
            # python -W ignore mimiciv_tasks.py  --num_train_epochs 30 \
            #                 --kernel_size 1 --train_bs_mimic 8 --train_bs_eicu 128 --train_bs_embed 512 \
            #                 --eval_batch_size 8 --seed "${SEED}" \
            #                 --gradient_accumulation_steps 16  --num_update_bert_epochs 2 --bertcount 0 \
            #                 --ts_learning_rate 0.0004 --txt_learning_rate 0.00002 \
            #                 --notes_order 'Last' --num_of_notes 5 --max_length 1024 --layers 3\
            #                 --embed_dim 128 \
            #                 --perceiver_dim 64 \
            #                 --model_name "bioLongformer"\
            #                 --task "${TASK}"\
            #                 --ihm_mod 'TS-Text'\
            #                 --los_mod 'TS-CXR'\
            #                 --pheno_mod 'TS-Text-CXR'\
            #                 --rad_mod 'T1-T2-T3-T4-T5'\
            #                 --mor_mod 'T1-T2-T3-T4-T5'\
            #                 --birads_mod 'cc-mlo-2dcc-2dmlo'\
            #                 --risk_mod 'cc-mlo-2dcc-2dmlo'\
            #                 --density_mod 'cc-mlo-2dcc-2dmlo'\
            #                 --diag_mod 'I-G-C-B' \
            #                 --mimic_path '/export/io79/data/schaud35/datasets/'\
            #                 --eicu_path '/export/io79/data/schaud35/datasets/eicu/processed/'\
            #                 --embed_path '/export/io79/data/schaud35/datasets/EMBED'\
            #                 --adni_path '/export/io79/data/schaud35/datasets/adni/adni_processed/'\
            #                 --num_heads 8\
            #                 --embed_time 64\
            #                 --tt_max 48\
            #                 --tt_max_eicu 1\
            #                 --TS_mixup\
            #                 --mixup_level 'batch'\
            #                 --cross_method 'moe'\
            #                 --fp16 \
            #                 --reg_ts \
            #                 --balance_loss_coef 0.01 \
            #                 --fusion_model 'flexmoe' \
            #                 --num_of_experts "${EXPERT}" \
            #                 --top_k 2 4 \
            #                 --router_type 'joint' \
            #                 --gating_function "${GATING}" \
            #                 --use_pt_text_embeddings \
            #                 --shared_modality_encoders \
            #                 --modality_drop_rate 0.0
        # done
        RESULT_DIR="${RESULTS_DIR}/${FUSION_MODEL}/singletask/${GATING}/${TASK}/mod_drop_rate_${MOD_DROP_RATE}/experts_${EXPERT}"
        python aggregate_results.py --result_dir "${RESULT_DIR}"
    done
done


# For multi-task MIMIC use train_bs_mimic=2 else 4
# For single task without --use_pt_text_embeddings use train_bs_mimic=8, num_of_notes=5