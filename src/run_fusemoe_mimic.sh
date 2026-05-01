export CUDA_VISIBLE_DEVICES=4

python -W ignore mimiciv_tasks.py  --num_train_epochs 50 \
                --kernel_size 1 --train_bs_mimic 8 --train_bs_eicu 128 --train_bs_embed 512 \
                --eval_batch_size 8 --seed 42 \
                --gradient_accumulation_steps 16  --num_update_bert_epochs 2 --bertcount 0 \
                --ts_learning_rate 0.0004 --txt_learning_rate 0.00002 \
                --notes_order 'Last' --num_of_notes 5 --max_length 1024 --layers 3\
                --embed_dim 128 \
                --perceiver_dim 128 \
                --model_name "bioLongformer"\
                --task 'pheno'\
                --ihm_mod 'Text-CXR'\
                --los_mod 'TS-Text'\
                --pheno_mod 'Text-CXR'\
                --rad_mod 'T1-T2-T3-T4-T5'\
                --mor_mod 'T1-T2-T3-T4-T5'\
                --birads_mod 'cc-mlo-2dcc-2dmlo'\
                --risk_mod 'cc-mlo-2dcc-2dmlo'\
                --density_mod 'cc-mlo-2dcc-2dmlo'\
                --mimic_path '/export/io79/data/schaud35/datasets/'\
                --eicu_path '/export/io79/data/schaud35/datasets/eicu/processed/'\
                --embed_path '/export/io79/data/schaud35/datasets/EMBED/'\
                --num_heads 8\
                --embed_time 64\
                --tt_max 48\
                --tt_max_eicu 1\
                --TS_mixup\
                --mixup_level 'batch'\
                --cross_method 'moe'\
                --fp16 \
                --reg_ts \
                --fusion_model 'fusemoe' \
                --use_pt_text_embeddings \
                --num_of_experts 3 5 \
                --top_k 2 4 \
                --router_type 'joint' \
                --gating_function "laplace" \
                --shared_modality_encoders \
<<<<<<< HEAD
                --results_dir '/cis/home/xhan56/code/clinical-highmmt/src/results' \
=======
                --modality_drop_rate 0.0 \
                # --multitask_moe \
                # --use_wandb \
                # --wandb_project 'clinical-highmmt' 
>>>>>>> continual-learning
                # --linear_probe \
                # --base_task_mods 'TS-Text-CXR' \
                # --base_task 'los' \
                # --use_pt_text_embeddings \


# For multi-task MIMIC use train_bs_mimic=2 else 4
# For single task without --use_pt_text_embeddings use train_bs_mimic=8, num_of_notes=5