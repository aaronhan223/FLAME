import torch
from src.eval_scripts.performance import metrics_multilabel
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from tqdm import tqdm
import numpy as np
import pdb
from peft import LoraConfig, get_peft_model, TaskType
import os
from src.utils import *

def train(
    model,
    trains,
    valid,
    test,
    modalities,
    savedir,
    args,
    encoder,
    setting,
    criterion,
    lr=0.001,
    weight_decay=0.0, 
    optimizer=torch.optim.Adam, 
    device="cuda:0",
    train_weights=[1.0, 1.0],
    recon=False, 
    recon_weight=1, 
    recon_criterion=torch.nn.MSELoss(),
    flips=-1, 
    classesnum=[2,2,25],
    start_from=0,
    getattentionmap=False
    ):

    # Collect all parameters to optimize: model + all encoders
    if args.lora:
        # For LoRA: only trainable parameters from model + all encoder parameters
        params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))
        for enc in encoder.values():
            params_to_optimize += list(enc.parameters())
        optim = optimizer(params_to_optimize, lr=lr, weight_decay=weight_decay)
    else:
        # For full fine-tuning: all model parameters + all encoder parameters
        params_to_optimize = list(model.parameters())
        for enc in encoder.values():
            params_to_optimize += list(enc.parameters())
        optim = optimizer(params_to_optimize, lr=lr, weight_decay=weight_decay)
    # --- LoRA Setup ---
    # lora_config = LoraConfig(
    #     task_type=TaskType.FEATURE_EXTRACTION,  # or SEQ_CLS, CAUSAL_LM, etc. depending on model type
    #     r=8,           # rank of LoRA matrices
    #     lora_alpha=32, # scaling factor
    #     lora_dropout=0.1,
    #     target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],  # modify to match your model
    # )

    # # wrap your model
    # model = get_peft_model(model, lora_config)
    # print("Trainable parameters with LoRA:")
    # model.print_trainable_parameters()

    

    if args.num_train_epochs > 0:
        bestacc=0.0
        fulltrains=[]
        print('\nData preprocessing...')
        for i in tqdm(range(len(trains))): # for each task
            count=0 # count is batch number  
            for j in tqdm(trains[i]): # for each batch of that task
                # first round establish all the dictionaries for task 1, one per batch
                # second round utilize existing dictionaries for the next task
                # so fulltrains contain a list of dicts where the elements are divided by batch, each dict contains task and corresponding modalities of that batch
                if count >= len(fulltrains):
                    fulltrains.append({})
                fulltrains[count][str(i)] = j # a list of dicts, where keys are tasks, values are corresponding modality component of these tasks
                if i == flips:
                    j[-1] = (j[-1] + 1) % classesnum[i]
                count += 1
    # import pdb; pdb.set_trace()
    # fulltrains.reverse()
    # fulltrains=fulltrains[start_from:]
    task_names = {'MOR': 'mortality', 'RAD': 'readmission'}
    for ep in range(args.num_train_epochs):
        # if ep==0:
            # snapshot_before = check_encoder_updates(encoder, "Before optim.step()")
        model.train()
        for enc in encoder.values():
            enc.train()
        print(f'\nTraining epoch {ep}/{args.num_train_epochs}...')
        for js in tqdm(fulltrains): # for each sample
            optim.zero_grad()
            losses=0.0
            for ii in js: # for each task
                task = modalities[int(ii)][0].split('_')[1]
                if task in ['IHM', 'PHENO', 'LOS']:
                    ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts, input_ids_sequences, attn_mask_sequences, text_emb, note_time, note_time_mask, cxr_feats, cxr_time, cxr_time_mask, ecg_feats, ecg_time, ecg_time_mask, label, cxr_missing, text_missing, ecg_missing = js[ii]
                    # MIMIC-IV encoders
                    embeddings = encoder[task](
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
                        ecg_time_mask=ecg_time_mask, labels=label, reg_ts=reg_ts,\
                        cxr_missing=cxr_missing, text_missing=text_missing, ecg_missing=ecg_missing, modalities=modalities[int(ii)]
                    )
                elif task in ['MOR', 'RAD']:
                    # TODO: work on eicu only for perceiver first, use each type as a modality, does it make sense for different separation category for each dataset?
                    # keep encoders, replace classifiers with perceivers
                    codes, types, timestamps, ages, genders, ethnicities, label = js[ii]['codes'], js[ii]['types'], js[ii]['timestamps'], js[ii]['age'], js[ii]['gender'], js[ii]['ethnicity'], js[ii][task_names[task]].long()
                    embeddings = encoder[task](
                        codes=codes,
                        types=types,
                        timestamps=timestamps,
                        ages=ages,
                        genders=genders,
                        ethnicities=ethnicities,
                        modalities=modalities[int(ii)]
                    )
                if args.lora:
                    model.base_model.model.model.to_logits = model.base_model.model.model.to_logitslist[int(ii)]
                else:
                    model.to_logits = model.to_logitslist[int(ii)]
                indict={}
                for i in range(len(modalities[int(ii)])):
                    indict[modalities[int(ii)][i]] = embeddings[modalities[int(ii)][i]].float().to(device)
                
                if recon:
                    out, rec = model(indict=indict, task=task, use_recon=True) if args.lora else model(indict, task=task, use_recon=True)
                    stuffs = []
                    for modal in indict:
                        stuffs.append(torch.mean(indict[modal], dim=1))
                    origs = torch.cat(stuffs, dim=1)
                    loss = criterion[int(ii)](out, label.to(device)) + recon_weight * recon_criterion(rec, origs)
                else:
                    # import pdb; pdb.set_trace()
                    out = model(indict=indict, task=task) if args.lora else model(indict, task=task)
                    if 'TS_PHENO' in modalities[int(ii)] or 'Text_PHENO' in modalities[int(ii)] or 'CXR_PHENO' in modalities[int(ii)]:
                        loss=criterion[int(ii)](out, label.float().to(device))
                    else:
                        loss=criterion[int(ii)](out, label.to(device))
                losses += loss * train_weights[int(ii)]
            
            losses.backward()
            optim.step()
            # torch.cuda.empty_cache()
            # snapshot_after = check_encoder_updates(encoder, "After optim.step()")
            # compare_encoder_snapshots(snapshot_before, snapshot_after)

        with torch.no_grad():
            model.eval()
            for enc in encoder.values():
                enc.eval()
            accs=0.0
            eval_vals={}
            print("\nValidation...")
            for ii in tqdm(range(len(valid))): # for each task
                task = modalities[int(ii)][0].split('_')[1]
                eval_logits = []
                eval_labels = []
                for jj in tqdm(valid[ii]): # for each sample
                    if task in ['IHM', 'PHENO', 'LOS']:
                        ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts, input_ids_sequences, attn_mask_sequences, text_emb, note_time, note_time_mask, cxr_feats, cxr_time, cxr_time_mask, ecg_feats, ecg_time, ecg_time_mask, label, cxr_missing, text_missing, ecg_missing = jj
                        embeddings = encoder[task](
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
                            cxr_missing=cxr_missing, text_missing=text_missing, ecg_missing=ecg_missing, modalities=modalities[int(ii)]
                        )
                    elif task in ['MOR', 'RAD']:
                        codes, types, timestamps, ages, genders, ethnicities, label = jj['codes'], jj['types'], jj['timestamps'], jj['age'], jj['gender'], jj['ethnicity'], jj[task_names[task]].long()
                        embeddings = encoder[task](
                            codes=codes,
                            types=types,
                            timestamps=timestamps,
                            ages=ages,
                            genders=genders,
                            ethnicities=ethnicities,
                            modalities=modalities[int(ii)]
                        )
                    if args.lora:
                        model.base_model.model.model.to_logits = model.base_model.model.model.to_logitslist[ii]
                    else:
                        model.to_logits = model.to_logitslist[ii]
                    indict={}
                    for i in range(len(modalities[ii])):
                        indict[modalities[ii][i]] = embeddings[modalities[ii][i]].float().to(device)
                    out = model(indict=indict, task=task) if args.lora else model(indict, task=task)
                    if 'TS_PHENO' in modalities[int(ii)] or 'Text_PHENO' in modalities[int(ii)] or 'CXR_PHENO' in modalities[int(ii)]:
                        logit = torch.nn.functional.sigmoid(out)
                    else:
                        logit = torch.nn.functional.softmax(out, dim=-1)[:, 1]
                    logit = logit.cpu().numpy()
                    label_ids = label.cpu().numpy()
                    eval_logits += logit.tolist()
                    eval_labels += label_ids.tolist()

                all_logits = np.array(eval_logits)
                all_label = np.array(eval_labels)
                # use auc score as picking best performing model metric
                if 'TS_PHENO' in modalities[int(ii)] or 'Text_PHENO' in modalities[int(ii)] or 'CXR_PHENO' in modalities[int(ii)]:
                    eval_vals = metrics_multilabel(all_label, all_logits, verbose=0)
                    accs += eval_vals['auc_scores'].mean()
                else:
                    eval_val = roc_auc_score(all_label, all_logits)
                    accs += eval_val

            if accs > bestacc:
                bestacc = accs
                # torch.save(model, savedir)
                for ii in range(len(modalities)):
                    task = modalities[int(ii)][0].split('_')[1]
                    # torch.save(encoder[task], f'{savedir.split(".pt")[0]}_{task}_encoder.pt')
        print('Model saved to ', savedir)
        # import pdb; pdb.set_trace()
        ### Testing function ###
        # model=torch.load(savedir).to(device)
        if (ep+1)%5==0 or ep==args.num_train_epochs-1:    
            model.eval()
            for enc_k, enc in encoder.items():
                # encoder[enc_k]=torch.load(f'{savedir.split(".pt")[0]}_{enc_k}_encoder.pt').to(device)
                encoder[enc_k].eval()
            testaccs=[]
            with torch.no_grad():
                rets=[[],[],[],[]]
                print("\nTest...")
                task_mods_dict = {
                    'ihm_mod': args.ihm_mod,
                    'los_mod': args.los_mod,
                    'pheno_mod': args.pheno_mod,
                    'ihm-los-pheno_mod': args.ihm_mod+'_'+args.los_mod+'_'+args.pheno_mod,
                    'ihm-los_mod': args.ihm_mod+'_'+args.los_mod,
                    'ihm-pheno_mod': args.ihm_mod+'_'+args.pheno_mod,
                    'los-pheno_mod': args.los_mod+'_'+args.pheno_mod,
                    'readmission_mod': args.rad_mod,
                    'mortality_mod': args.mor_mod,
                    'ihm-mortality_mod': args.ihm_mod+'_'+args.mor_mod,
                    'los-readmission_mod': args.los_mod+'_'+args.rad_mod
                }
                task_mod_key = f'{args.task}_mod'
                if args.lora:
                    out_fname = f"{args.results_dir}/{args.fusion_model}/{args.base_task}/{args.base_task_mods}/result_{args.task}_{task_mods_dict[task_mod_key]}_lora_from_{args.base_task}.txt"
                    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
                    f = open(out_fname, 'a')
                elif args.fine_tune:
                    out_fname = f"{args.results_dir}/{args.fusion_model}/{args.base_task}/{args.base_task_mods}/result_{args.task}_{task_mods_dict[task_mod_key]}_ft_from_{args.base_task}.txt"
                    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
                    f = open(out_fname, 'a')
                elif args.linear_probe:
                    out_fname = f"{args.results_dir}/{args.fusion_model}/{args.base_task}/{args.base_task_mods}/result_{args.task}_{task_mods_dict[task_mod_key]}_linear_probe_from_{args.base_task}.txt"
                    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
                    f = open(out_fname, 'a')
                else:
                    if args.shared_modality_encoders:
                        out_fname = f"{args.results_dir}/{args.fusion_model}/multitask/{args.task}/result_{args.task}_{task_mods_dict[task_mod_key]}.txt"
                    else:
                        out_fname = f"{args.results_dir}/{args.fusion_model}/{args.base_task}/{args.base_task_mods}/result_{args.task}_{task_mods_dict[task_mod_key]}.txt"
                    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
                    f = open(out_fname, 'a')
                f.write(f"\n################## Epoch {ep} ##################\n")
                f.write(setting + "  \n")
                print(f"\nWriting results to {out_fname}\n")
                # import pdb; pdb.set_trace()
                for ii in tqdm(range(len(test))):
                    eval_vals={}
                    eval_logits = []
                    eval_labels = []
                    task = modalities[int(ii)][0].split('_')[1]
                    
                    if args.lora:
                        model.base_model.model.model.to_logits = model.base_model.model.model.to_logitslist[ii]
                    else:
                        model.to_logits=model.to_logitslist[ii]
                    for jj in tqdm(test[ii]):
                        if task in ['IHM', 'PHENO', 'LOS']:
                            ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts, input_ids_sequences, attn_mask_sequences, text_emb, note_time, note_time_mask, cxr_feats, cxr_time, cxr_time_mask, ecg_feats, ecg_time, ecg_time_mask, label, cxr_missing, text_missing, ecg_missing = jj
                            embeddings = encoder[task](
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
                                cxr_missing=cxr_missing, text_missing=text_missing, ecg_missing=ecg_missing, modalities=modalities[int(ii)]
                            )
                        elif task in ['MOR', 'RAD']:
                            codes, types, timestamps, ages, genders, ethnicities, label = jj['codes'], jj['types'], jj['timestamps'], jj['age'], jj['gender'], jj['ethnicity'], jj[task_names[task]].long()
                            embeddings = encoder[task](
                                codes=codes,
                                types=types,
                                timestamps=timestamps,
                                ages=ages,
                                genders=genders,
                                ethnicities=ethnicities,
                                modalities=modalities[int(ii)]
                            )
                        indict={}
                        for i in range(0, len(modalities[ii])): # for each modality within that task
                            indict[modalities[ii][i]] = embeddings[modalities[ii][i]].float().to(device)
                        
                        out = model(indict=indict, task=task) if args.lora else model(indict, task=task)
                        if 'TS_PHENO' in modalities[int(ii)] or 'Text_PHENO' in modalities[int(ii)] or 'CXR_PHENO' in modalities[int(ii)]:
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
                    if 'TS_PHENO' in modalities[int(ii)] or 'Text_PHENO' in modalities[int(ii)] or 'CXR_PHENO' in modalities[int(ii)]:
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
