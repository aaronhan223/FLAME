export CUDA_VISIBLE_DEVICES=1

python -W ignore mimiciv_tasks.py  --num_train_epochs 2 \
                --kernel_size 1 --train_batch_size 8 --eval_batch_size 8 --seed 42 \
                --gradient_accumulation_steps 16  --num_update_bert_epochs 2 --bertcount 0 \
                --ts_learning_rate 0.0004 --txt_learning_rate 0.00002 \
                --notes_order 'Last' --num_of_notes 5 --max_length 1024 --layers 3\
                --embed_dim 128 \
                --perceiver_dim 64 \
                --model_name "bioLongformer"\
                --task 'ihm-pheno'\
                --ihm_mod 'TS-Text-CXR'\
                --los_mod ''\
                --pheno_mod 'TS-Text-CXR'\
                --file_path '/cis/home/xhan56/code/Multimodal-Transformer/src/Data/'\
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
                --reg_ts