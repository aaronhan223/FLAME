import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import (AutoTokenizer,
                          AutoModel,
                          AutoConfig,
                          AdamW,
                          BertTokenizer,
                          BertModel,
                          get_scheduler,
                          set_seed,
                          BertPreTrainedModel,
                          LongformerConfig,
                          LongformerModel,
                          LongformerTokenizer,
                         )
from src.crossattnperceiver import MultiModalityPerceiver, InputModality, PerceiverWrapper
from src.mimiciv_tasks import loadBert
from src.train_structure_multitask_mimic import train
from src.encoders import ModalityEncoders, FSEncoder
from src.utils import create_directory, dump_pickle
from src.preprocess.preprocess_eicu import *
import torch
from accelerate import Accelerator
torch.multiprocessing.set_sharing_strategy('file_system')
from src.datasets.mimic.get_data_mimic_iv import data_prepare as prepare_mimic
from src.get_data_eicu import data_prepare as prepare_eicu
from src.eval_scripts.performance import metrics_multilabel
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from tqdm import tqdm
import numpy as np
import pdb
from peft import LoraConfig, get_peft_model, TaskType


def evaluate_model(args, model, encoder, device, getattentionmap=False):
    modalities = set()
    if len(args.ihm_mod) != 0 and 'ihm' in args.task:
        for e in args.ihm_mod.split("-"):
            modalities.add(e)
    if len(args.los_mod) != 0 and 'los' in args.task:
        for e in args.los_mod.split("-"):
            modalities.add(e)
    if len(args.pheno_mod) != 0 and 'pheno' in args.task:
        for e in args.pheno_mod.split("-"):
            modalities.add(e)
    modeltype = ''
    modals = [*modalities]
    modals.sort()
    for m in modals:
        modeltype = modeltype + m + '_'
    modeltype = modeltype[:-1]

    if len(args.rad_mod) != 0 and 'readmission' in args.task:
        for e in args.rad_mod.split("-"):
            modalities.add(e)
    if len(args.mor_mod) != 0 and 'mortality' in args.task:
        for e in args.mor_mod.split("-"):
            modalities.add(e)
    modalities = [*modalities]

    if 'Text' in modeltype:
        BioBert, BioBertConfig, tokenizer = loadBert(args, device)
    else:
        tokenizer = None
        BioBert = None

    tasks = args.task.split("-")
    train = []
    valid = []
    test = []
    modalities_per_task = []
    
    if 'ihm' in tasks:
        train_ihm, valid_ihm, test_ihm = prepare_mimic(args=args, task='ihm', tokenizer=tokenizer, modeltype=modeltype)
        train.append(train_ihm)
        valid.append(valid_ihm)
        test.append(test_ihm)
        ihm_mods = list(map(lambda s: s + '_IHM', args.ihm_mod.split("-")))
        assert len(ihm_mods) > 1, "At least two modalities per task!"
        modalities_per_task.append(ihm_mods)
    
    if 'los' in tasks:
        train_los, valid_los, test_los = prepare_mimic(args=args, task='los', tokenizer=tokenizer, modeltype=modeltype)
        train.append(train_los)
        valid.append(valid_los)
        test.append(test_los)
        los_mods = list(map(lambda s: s + '_LOS', args.los_mod.split("-")))
        assert len(los_mods) > 1, "At least two modalities per task!"
        modalities_per_task.append(los_mods)
    
    if 'pheno' in tasks:
        train_pheno, valid_pheno, test_pheno = prepare_mimic(args=args, task='pheno', tokenizer=tokenizer, modeltype=modeltype)
        train.append(train_pheno)
        valid.append(valid_pheno)
        test.append(test_pheno)
        pheno_mods = list(map(lambda s: s + '_PHENO', args.pheno_mod.split("-")))
        assert len(pheno_mods) > 1, "At least two modalities per task!"
        modalities_per_task.append(pheno_mods)
    
    if 'readmission' in tasks:
        train_rad, valid_rad, test_rad, tokenizer_rad = prepare_eicu(args=args)
        train.append(train_rad)
        valid.append(valid_rad)
        test.append(test_rad)
        rad_mods = list(map(lambda s: s + '_RAD', args.rad_mod.split("-")))
        modalities_per_task.append(rad_mods)
    
    if 'mortality' in tasks:
        train_mor, valid_mor, test_mor, tokenizer_mor = prepare_eicu(args=args)
        train.append(train_mor)
        valid.append(valid_mor)
        test.append(test_mor)
        mor_mods = list(map(lambda s: s + '_MOR', args.mor_mod.split("-")))
        modalities_per_task.append(mor_mods)
        
    setting = '{}-{}-seed{}-Mbs{}-Ebs{}-ep{}-enc_head{}-embd_dim{}-perceiver_dim{}-ttmax{}-embd_time{}-{}'.format(
        args.task,
        modeltype,
        args.seed,
        args.train_bs_mimic,
        args.train_bs_eicu,
        args.num_train_epochs,
        args.num_heads,
        args.embed_dim,
        args.perceiver_dim,
        args.tt_max,
        args.embed_time,
        modalities_per_task
    )
    
    ### Testing function ###
    model.eval()
    encoder.eval()
    testaccs=[]
    with torch.no_grad():
        rets=[[],[],[],[]]
        print("\nTest...")
        task_mods_dict = {
            'ihm_mod': args.ihm_mod,
            'los_mod': args.los_mod,
            'pheno_mod': args.pheno_mod,
            'rad_mod': args.rad_mod,
            'mor_mod': args.mor_mod
        }
        task_mod_key = f'{args.task}_mod'
        if args.lora:
            out_fname = f"{args.results_dir}/results_merged/{args.base_task}/{args.base_task_mods}/result_{args.task}_{args.new_task_mods}_lora_from_ihm.txt"
            os.makedirs(os.path.dirname(out_fname), exist_ok=True)
            f = open(out_fname, 'a')
        elif args.fine_tune:
            out_fname = f"{args.results_dir}/results_merged/{args.base_task}/{args.base_task_mods}/result_{args.task}_{args.new_task_mods}_ft_from_ihm.txt"
            os.makedirs(os.path.dirname(out_fname), exist_ok=True)
            f = open(out_fname, 'a')
        else:
            out_fname = f"{args.results_dir}/results_merged/{args.base_task}/{args.base_task_mods}/result_{args.task}_{args.new_task_mods}.txt"
            os.makedirs(os.path.dirname(out_fname), exist_ok=True)
            f = open(out_fname, 'a')
        f.write(f"\n################## New Experiment ##################\n")
        f.write(setting + "  \n")
        print(f"\nWriting results to {out_fname}\n")
        for ii in tqdm(range(len(test))):
            eval_vals={}
            eval_logits = []
            eval_labels = []
            task = modalities_per_task[int(ii)][0].split('_')[1]
            
            if args.lora:
                model.base_model.model.model.to_logits = model.base_model.model.model.to_logitslist[ii]
            else:
                model.to_logits=model.to_logitslist[ii]
            for jj in tqdm(test[ii]):
                if task in ['IHM', 'PHENO', 'LOS']:
                    ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts, input_ids_sequences, attn_mask_sequences, text_emb, note_time, note_time_mask, cxr_feats, cxr_time, cxr_time_mask, ecg_feats, ecg_time, ecg_time_mask, label, cxr_missing, text_missing, ecg_missing = jj
                    embeddings = encoder(
                        x_ts=ts_input_sequences, \
                        x_ts_mask=ts_mask_sequences,\
                        ts_tt_list=ts_tt,\
                        input_ids_sequences=input_ids_sequences,\
                        attn_mask_sequences=attn_mask_sequences, text_emb=text_emb, note_time_list=note_time,\
                        note_time_mask_list=note_time_mask,\
                        cxr_feats=cxr_feats,\
                        cxr_time=cxr_time, \
                        cxr_time_mask=cxr_time_mask,\
                        ecg_feats=ecg_feats,\
                        ecg_time=ecg_time, \
                        ecg_time_mask=ecg_time_mask,labels=label,reg_ts=reg_ts,\
                        cxr_missing=cxr_missing, text_missing=text_missing, ecg_missing=ecg_missing, modalities=modalities_per_task[int(ii)]
                    )
                elif task in ['MOR', 'RAD']:
                    codes, types, timestamps, ages, genders, ethnicities, label = jj['codes'], jj['types'], jj['timestamps'], jj['age'], jj['gender'], jj['ethnicity'], jj[task_names[task]].long()
                    embeddings = encoder(
                        codes=codes,
                        types=types,
                        timestamps=timestamps,
                        ages=ages,
                        genders=genders,
                        ethnicities=ethnicities,
                        modalities=modalities_per_task[int(ii)]
                    )
                indict={}
                for i in range(0, len(modalities_per_task[int(ii)])): # for each modality within that task
                    indict[modalities_per_task[int(ii)][i]] = embeddings[modalities_per_task[int(ii)][i]].float().to(device)
                out = model(indict=indict) if args.lora else model(indict)
                if 'TS_PHENO' in modalities_per_task[int(ii)]:
                    logit = torch.nn.functional.sigmoid(out)
                else:
                    logit = torch.nn.functional.softmax(out, dim=-1)[:, 1]
                logits = logit.cpu().numpy()
                labels = label.cpu().numpy()
                eval_logits += logits.tolist()
                eval_labels += labels.tolist()
                if getattentionmap:
                    rets[ii].append(model.attns)
            all_logits = np.array(eval_logits)
            all_label = np.array(eval_labels)
            all_pred = np.where(all_logits > 0.5, 1, 0)
            if 'TS_PHENO' in modalities_per_task[int(ii)]:
                eval_vals = metrics_multilabel(all_label, all_logits, verbose=0)
                eval_vals['macro_f1'] = f1_score(all_label, all_pred, average='macro')
            else:
                eval_val = roc_auc_score(np.array(eval_labels), np.array(eval_logits))
                eval_vals['auc'] = eval_val
                (precisions, recalls, thresholds) = precision_recall_curve(np.array(eval_labels), np.array(eval_logits))
                eval_val = auc(recalls, precisions)
                eval_vals['auprc'] = eval_val
                eval_val = f1_score(np.array(eval_labels), all_pred)
                eval_vals['f1'] = eval_val
            
            f.write(f"------Task {ii}------\n")
            for k, v in eval_vals.items():
                f.write(k+': {}'.format(v))
                f.write('\n')
                f.write('\n')
        f.close()
    
    if getattentionmap:
        return rets
    return testaccs