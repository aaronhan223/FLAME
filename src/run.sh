export CUDA_VISIBLE_DEVICES=6

# python -W ignore mimiciv_tasks.py  --num_train_epochs 50 \
#                 --kernel_size 1 --train_bs_mimic 8 --train_bs_eicu 128 \
#                 --eval_batch_size 8 --seed 42 \
#                 --gradient_accumulation_steps 16  --num_update_bert_epochs 2 --bertcount 0 \
#                 --ts_learning_rate 0.0004 --txt_learning_rate 0.00002 \
#                 --notes_order 'Last' --num_of_notes 5 --max_length 1024 --layers 3\
#                 --embed_dim 128 \
#                 --perceiver_dim 64 \
#                 --model_name "bioLongformer"\
#                 --task 'ihm-los-pheno-mortality-readmission'\
#                 --ihm_mod 'TS-Text-CXR'\
#                 --los_mod 'TS-CXR'\
#                 --pheno_mod 'TS-CXR'\
#                 --rad_mod 'T1-T2-T3-T4-T5'\
#                 --mor_mod 'T1-T2-T3-T4-T5'\
#                 --mimic_path '/export/io79/data/schaud35/'\
#                 --eicu_path '/export/io79/data/schaud35/eicu/processed/'\
#                 --num_heads 8\
#                 --embed_time 64\
#                 --tt_max 48\
#                 --TS_mixup\
#                 --fp16 \
#                 --irregular_learn_emb_text \
#                 --irregular_learn_emb_ts \
#                 --irregular_learn_emb_cxr \
#                 --irregular_learn_emb_ecg \
#                 --use_pt_text_embeddings \
#                 --reg_ts

# Continual Learning - Stage 1
python -W ignore mimiciv_tasks.py  --num_train_epochs 50 \
                --kernel_size 1 --train_bs_mimic 8 --train_bs_eicu 128 \
                --eval_batch_size 8 --seed 42 \
                --gradient_accumulation_steps 16  --num_update_bert_epochs 2 --bertcount 0 \
                --ts_learning_rate 0.0004 --txt_learning_rate 0.00002 \
                --notes_order 'Last' --num_of_notes 5 --max_length 1024 --layers 3\
                --embed_dim 128 \
                --perceiver_dim 64 \
                --model_name "bioLongformer"\
                --task 'ihm-pheno'\
                --ihm_mod 'TS-Text-CXR'\
                --los_mod 'TS-Text-CXR'\
                --pheno_mod 'Text-CXR'\
                --rad_mod 'T1-T2-T3-T4-T5'\
                --mor_mod 'T1-T2-T3-T4-T5'\
                --mimic_path '/export/io79/data/schaud35/'\
                --eicu_path '/export/io79/data/schaud35/eicu/processed/'\
                --num_heads 8\
                --embed_time 64\
                --tt_max 48\
                --TS_mixup\
                --fp16 \
                --irregular_learn_emb_text \
                --irregular_learn_emb_ts \
                --irregular_learn_emb_cxr \
                --irregular_learn_emb_ecg \
                --use_pt_text_embeddings \
                --reg_ts \
                --fusion_model 'crossattntransformer' \
                --shared_modality_encoders
                # --linear_probe \
                # --base_task_mods 'TS-Text-CXR' \
                # --base_task 'los' \