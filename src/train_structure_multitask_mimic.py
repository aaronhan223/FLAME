import torch
from src.eval_scripts.performance import metrics_multilabel, metrics_multiclass
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
import random
import pdb
from peft import LoraConfig, get_peft_model, TaskType
import os
from src.utils import *

try:
    import wandb
except ImportError:
    wandb = None


def drop_modalities(indict, drop_rate):
    """Randomly mask modalities from indict with zeros. Always keeps at least one modality active.

    Args:
        indict: dict mapping modality names to their embedding tensors.
        drop_rate: probability of dropping each modality (0.0 = no dropping).

    Returns:
        A copy of indict with dropped modalities replaced by zero tensors of the same shape.
    """
    if drop_rate <= 0 or len(indict) <= 1:
        return indict
    keys = list(indict.keys())
    keep = [k for k in keys if random.random() >= drop_rate]
    # Ensure at least one modality remains active
    if len(keep) == 0:
        keep = [random.choice(keys)]

    masked = {}
    for key in keys:
        if key in keep:
            masked[key] = indict[key]
        else:
            masked[key] = torch.zeros_like(indict[key])
    return masked


def _to_float_if_scalar(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if np.isscalar(value):
        return float(value)
    return None


def _grad_l2_norm(parameters):
    total_sq_norm = 0.0
    has_grad = False
    for param in parameters:
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().data.norm(2).item()
        total_sq_norm += grad_norm ** 2
        has_grad = True
    if not has_grad:
        return 0.0
    return total_sq_norm ** 0.5


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

    model_grad_params = [p for p in model.parameters() if p.requires_grad]
    encoder_grad_params = []
    for enc in encoder.values():
        encoder_grad_params.extend([p for p in enc.parameters() if p.requires_grad])

    use_wandb = bool(getattr(args, 'use_wandb', False) or getattr(args, 'wandb', False))
    wandb_run_started_here = False
    if use_wandb and wandb is None:
        print("[warn] wandb logging requested but wandb is not installed. Continuing without wandb.")
        use_wandb = False
    if use_wandb and wandb.run is None:
        wandb.init(
            entity='shravan25-jhu',
            project=getattr(args, 'wandb_project', 'clinical-highmmt'),
            name=getattr(args, 'wandb_run_name', f'{args.fusion_model}_{args.task}_{"_".join([getattr(args, f"{t}_mod") for t in args.task.split("-")])}_multitask_run'),
            config=vars(args) if hasattr(args, '__dict__') else None
        )
        wandb_run_started_here = True

    for ep in range(args.num_train_epochs):
        
        # if ep==0:
            # snapshot_before = check_encoder_updates(encoder, "Before optim.step()")
        model.train()
        for enc in encoder.values():
            enc.train()
        print(f'\nTraining epoch {ep}/{args.num_train_epochs}...')
        epoch_train_loss_sum = 0.0
        epoch_model_grad_norm_sum = 0.0
        epoch_encoder_grad_norm_sum = 0.0
        epoch_train_steps = 0
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
                elif task.lower() in ['birads', 'risk', 'density']:
                    idx, label, embed_2dcc, embed_2dmlo, embed_cc, embed_mlo, all_views = js[ii]
                    embeddings = encoder[task](
                        embed_cc=embed_cc, embed_mlo=embed_mlo, embed_2dcc=embed_2dcc, embed_2dmlo=embed_2dmlo, all_views=all_views, modalities=modalities[int(ii)], task=task
                    )
                
                if args.lora:
                    model.base_model.model.model.to_logits = model.base_model.model.model.to_logitslist[int(ii)]
                else:
                    model.to_logits = model.to_logitslist[int(ii)]
                indict={}
                for i in range(len(modalities[int(ii)])):
                    indict[modalities[int(ii)][i]] = embeddings[modalities[int(ii)][i]].float().to(device)
                indict = drop_modalities(indict, args.modality_drop_rate)
                
                if recon:
                    out, rec = model(indict=indict, task=task, use_recon=True) if args.lora else model(indict, task=task, use_recon=True)
                    stuffs = []
                    for modal in indict:
                        stuffs.append(torch.mean(indict[modal], dim=1))
                    origs = torch.cat(stuffs, dim=1)
                    loss = criterion[int(ii)](out, label.to(device)) + recon_weight * recon_criterion(rec, origs)
                else:
                    out = model(indict=indict, task=task) if args.lora else model(indict, task=task)
                    if 'PHENO' in modalities[int(ii)][0]:
                        loss=criterion[int(ii)](out, label.float().to(device))
                    elif 'birads' in modalities[int(ii)][0].lower() or 'density' in modalities[int(ii)][0].lower():
                        loss=criterion[int(ii)](out, label.to(device))
                    else:
                        loss=criterion[int(ii)](out, label.to(device))
                losses += loss * train_weights[int(ii)]
            losses.backward()
            batch_model_grad_norm = _grad_l2_norm(model_grad_params)
            batch_encoder_grad_norm = _grad_l2_norm(encoder_grad_params)
            # total = 0.0
            # for p in model.parameters():
            #     if p.requires_grad and p.grad is not None:
            #         total += p.grad.data.norm(2).item()
            # print("grad_norm:", total)
            optim.step()
            epoch_train_loss_sum += losses.item()
            epoch_model_grad_norm_sum += batch_model_grad_norm
            epoch_encoder_grad_norm_sum += batch_encoder_grad_norm
            epoch_train_steps += 1
            # torch.cuda.empty_cache()
            # snapshot_after = check_encoder_updates(encoder, "After optim.step()")
            # compare_encoder_snapshots(snapshot_before, snapshot_after)

        train_log = {}
        if epoch_train_steps > 0:
            train_log['train/loss'] = epoch_train_loss_sum / epoch_train_steps
            train_log['train/grad_norm/model'] = epoch_model_grad_norm_sum / epoch_train_steps
            train_log['train/grad_norm/encoder'] = epoch_encoder_grad_norm_sum / epoch_train_steps

        with torch.no_grad():
            model.eval()
            for enc in encoder.values():
                enc.eval()
            accs=0.0
            eval_vals={}
            val_log = {}
            val_loss_total_sum = 0.0
            val_loss_total_steps = 0
            print("\nValidation...")
            for ii in tqdm(range(len(valid))): # for each task
                task = modalities[int(ii)][0].split('_')[1]
                eval_logits = []
                eval_labels = []
                val_loss_task_sum = 0.0
                val_loss_task_steps = 0
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
                    elif task.lower() in ['birads', 'risk', 'density']:
                        idx, label, embed_2dcc, embed_2dmlo, embed_cc, embed_mlo, all_views = jj
                        embeddings = encoder[task](
                            embed_cc=embed_cc, embed_mlo=embed_mlo, embed_2dcc=embed_2dcc, embed_2dmlo=embed_2dmlo, all_views=all_views, modalities=modalities[int(ii)], task=task
                        )
                    if args.lora:
                        model.base_model.model.model.to_logits = model.base_model.model.model.to_logitslist[ii]
                    else:
                        model.to_logits = model.to_logitslist[ii]
                    indict={}
                    for i in range(len(modalities[ii])):
                        indict[modalities[ii][i]] = embeddings[modalities[ii][i]].float().to(device)
                    indict = drop_modalities(indict, args.modality_drop_rate)
                    
                    if recon:
                        out, rec = model(indict=indict, task=task, use_recon=True) if args.lora else model(indict, task=task, use_recon=True)
                        stuffs = []
                        for modal in indict:
                            stuffs.append(torch.mean(indict[modal], dim=1))
                        origs = torch.cat(stuffs, dim=1)
                        val_loss = criterion[int(ii)](out, label.to(device)) + recon_weight * recon_criterion(rec, origs)
                    else:
                        out = model(indict=indict, task=task) if args.lora else model(indict, task=task)
                        if 'PHENO' in modalities[int(ii)][0]:
                            val_loss = criterion[int(ii)](out, label.float().to(device))
                        elif 'birads' in modalities[int(ii)][0].lower() or 'density' in modalities[int(ii)][0].lower():
                            val_loss = criterion[int(ii)](out, label.to(device))
                        else:
                            val_loss = criterion[int(ii)](out, label.to(device))
                    
                    val_loss_task_sum += val_loss.item()
                    val_loss_task_steps += 1
                    val_loss_total_sum += val_loss.item()
                    val_loss_total_steps += 1

                    if 'PHENO' in modalities[int(ii)][0]:
                        logit = torch.nn.functional.sigmoid(out)
                    elif 'birads' in modalities[int(ii)][0].lower() or 'density' in modalities[int(ii)][0].lower():
                        logit = torch.nn.functional.softmax(out, dim=-1)
                    else:
                        logit = torch.nn.functional.softmax(out, dim=-1)[:, 1]
                    logit = logit.cpu().numpy()
                    label_ids = label.cpu().numpy()
                    eval_logits += logit.tolist()
                    eval_labels += label_ids.tolist()

                all_logits = np.array(eval_logits)
                all_label = np.array(eval_labels)
                
                # use auc score as picking best performing model metric
                if 'PHENO' in modalities[int(ii)][0]:
                    eval_vals = metrics_multilabel(all_label, all_logits, verbose=0)
                    accs += eval_vals['auc_scores'].mean()
                    val_log[f'val/{task}/auc_mean'] = float(eval_vals['auc_scores'].mean())
                elif 'birads' in modalities[int(ii)][0].lower() or 'density' in modalities[int(ii)][0].lower():
                    eval_vals = metrics_multiclass(all_label, all_logits, verbose=0)
                    accs += eval_vals['ave_auc_macro']
                    val_log[f'val/{task}/ave_auc_macro'] = float(eval_vals['ave_auc_macro'])
                else:
                    eval_val = roc_auc_score(all_label, all_logits)
                    accs += eval_val
                    val_log[f'val/{task}/auc'] = float(eval_val)

                for metric_name, metric_val in eval_vals.items():
                    scalar_val = _to_float_if_scalar(metric_val)
                    if scalar_val is not None:
                        val_log[f'val/{task}/{metric_name}'] = scalar_val

                if val_loss_task_steps > 0:
                    val_log[f'val/{task}/loss'] = val_loss_task_sum / val_loss_task_steps

            val_log['val/score_sum'] = float(accs)
            val_log['val/best_score_sum'] = float(bestacc)
            if val_loss_total_steps > 0:
                val_log['val/loss'] = val_loss_total_sum / val_loss_total_steps

            if accs > bestacc:
                bestacc = accs
                # torch.save(model, savedir)
                for ii in range(len(modalities)):
                    task = modalities[int(ii)][0].split('_')[1]
                    # torch.save(encoder[task], f'{savedir.split(".pt")[0]}_{task}_encoder.pt')
                val_log['val/best_score_sum'] = float(bestacc)
        print('Model saved to ', savedir)
        if use_wandb:
            wandb.log({**train_log, **val_log, 'epoch': ep})
        # import pdb; pdb.set_trace()
        ### Testing function ###
        # model=torch.load(savedir).to(device)
        if (ep+1)%10==0 or ep==args.num_train_epochs-1:    
            model.eval()
            for enc_k, enc in encoder.items():
                # encoder[enc_k]=torch.load(f'{savedir.split(".pt")[0]}_{enc_k}_encoder.pt').to(device)
                encoder[enc_k].eval()
            testaccs=[]
            with torch.no_grad():
                rets=[[],[],[],[]]
                test_log = {}
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
                    'los-readmission_mod': args.los_mod+'_'+args.rad_mod,
                    'ihm-readmission_mod': args.ihm_mod+'_'+args.rad_mod,
                    'los-mortality_mod': args.los_mod+'_'+args.mor_mod,
                    'ihm-los-mortality_mod': args.ihm_mod+'_'+args.los_mod+'_'+args.mor_mod,
                    'ihm-los-mortality-readmission_mod': args.ihm_mod+'_'+args.los_mod+'_'+args.mor_mod+'_'+args.rad_mod,
                    'birads_mod': args.birads_mod,
                    'risk_mod': args.risk_mod,
                    'density_mod': args.density_mod,
                    'birads-risk-density_mod': args.birads_mod+'_'+args.risk_mod+'_'+args.density_mod,
                    'ihm-birads_mod': args.ihm_mod+'_'+args.birads_mod,
                    'birads-risk_mod': args.birads_mod+'_'+args.risk_mod,
                    'birads-density_mod': args.birads_mod+'_'+args.density_mod,
                    'risk-density_mod': args.risk_mod+'_'+args.density_mod,
                    'ihm-los-pheno-birads-risk-density_mod': args.ihm_mod+'_'+args.los_mod+'_'+args.pheno_mod+'_'+args.birads_mod+'_'+args.risk_mod+'_'+args.density_mod,
                    'ihm-los-pheno-mortality-readmission-birads-risk-density_mod': args.ihm_mod+'_'+args.los_mod+'_'+args.pheno_mod+'_'+args.mor_mod+'_'+args.rad_mod+'_'+args.birads_mod+'_'+args.risk_mod+'_'+args.density_mod
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
                        elif task.lower() in ['birads', 'risk', 'density']:
                            idx, label, embed_2dcc, embed_2dmlo, embed_cc, embed_mlo, all_views = jj
                            embeddings = encoder[task](
                                embed_cc=embed_cc, embed_mlo=embed_mlo, embed_2dcc=embed_2dcc, embed_2dmlo=embed_2dmlo, all_views=all_views, modalities=modalities[int(ii)], task=task
                            )
                        indict={}
                        for i in range(0, len(modalities[ii])): # for each modality within that task
                            indict[modalities[ii][i]] = embeddings[modalities[ii][i]].float().to(device)
                        indict = drop_modalities(indict, args.modality_drop_rate)
                        
                        out = model(indict=indict, task=task) if args.lora else model(indict, task=task)
                        if 'PHENO' in modalities[int(ii)][0]:
                            logit = torch.nn.functional.sigmoid(out)
                        elif 'birads' in modalities[int(ii)][0].lower() or 'density' in modalities[int(ii)][0].lower():
                            logit = torch.nn.functional.softmax(out, dim=-1)
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
                        
                    if 'PHENO' in modalities[int(ii)][0]:
                        all_pred = np.where(all_logits > 0.5, 1, 0)
                        eval_vals = metrics_multilabel(all_label, all_logits, verbose=0)
                        eval_vals['macro_f1'] = f1_score(all_label, all_pred, average='macro')
                        test_log[f'test/{task}/auc_mean'] = float(eval_vals['auc_scores'].mean())
                    elif 'birads' in modalities[int(ii)][0].lower() or 'density' in modalities[int(ii)][0].lower():
                        eval_vals = metrics_multiclass(all_label, all_logits, verbose=0)
                        all_pred = np.argmax(all_logits, axis=1)   # shape (N,)
                        print("label dist:", np.bincount(all_label.astype(int)))
                        print("pred dist :", np.bincount(all_pred.astype(int)))
                        eval_vals['macro_f1'] = f1_score(all_label, all_pred, average='macro')
                        eval_vals['accuracy'] = accuracy_score(all_label, all_pred)
                        test_log[f'test/{task}/ave_auc_macro'] = float(eval_vals['ave_auc_macro'])
                    else:
                        all_pred = np.where(all_logits > 0.5, 1, 0)
                        eval_val = roc_auc_score(np.array(eval_labels), np.array(eval_logits))
                        eval_vals['auc'] = eval_val
                        (precisions, recalls, thresholds) = precision_recall_curve(np.array(eval_labels), np.array(eval_logits))
                        eval_val = auc(recalls, precisions)
                        eval_vals['auprc'] = eval_val
                        eval_val = f1_score(np.array(eval_labels), all_pred)
                        eval_vals['f1'] = eval_val
                        eval_vals['accuracy'] = accuracy_score(all_label, all_pred)
                        test_log[f'test/{task}/auc'] = float(eval_vals['auc'])

                    for metric_name, metric_val in eval_vals.items():
                        scalar_val = _to_float_if_scalar(metric_val)
                        if scalar_val is not None:
                            test_log[f'test/{task}/{metric_name}'] = scalar_val
                    
                    f.write(f"------Task {ii}------\n")
                    for k, v in eval_vals.items():
                        f.write(k+': {}'.format(v))
                        f.write('\n')
                        f.write('\n')
                f.close()
                if use_wandb:
                    wandb.log({**test_log, 'epoch': ep})
    if use_wandb and wandb_run_started_here:
        wandb.finish()
    if getattentionmap:
        return rets
    return testaccs
