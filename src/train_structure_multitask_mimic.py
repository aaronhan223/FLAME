import torch
from src.eval_scripts.performance import metrics_multilabel, metrics_multiclass
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, hamming_loss
from tqdm import tqdm
import numpy as np
import random
import pdb
from peft import LoraConfig, get_peft_model, TaskType
import os
from src.utils import *
from src.analysis.moe_diagnostics import MoEDiagnosticsLogger, LayerwiseGradLogger

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
        return indict, {}
    keys = list(indict.keys())
    keep = [k for k in keys if random.random() >= drop_rate]
    # Ensure at least one modality remains active
    if len(keep) == 0:
        keep = [random.choice(keys)]

    masked = {}
    masked_keys = []
    for key in keys:
        if key in keep:
            masked[key] = indict[key]
        else:
            masked[key] = torch.zeros_like(indict[key])
            masked_keys.append(key)
    return masked, masked_keys


def replace_missing_embeddings(indict, missing_embeddings, masked_keys=[], optimizer=None):
    """Replace already-dropped (zeroed) modalities with learnable embeddings.

    Args:
        indict: dict mapping modality names to modality embedding tensors.
        missing_embeddings: torch.nn.ParameterDict storing one learnable token per modality.
        optimizer: optional optimizer; when provided, newly created embeddings are added to it.

    Returns:
        A copy of indict where zeroed modalities are replaced by their learned embeddings.
    """
    replaced = {}
    
    for key, value in indict.items():
        # is_dropped = bool(torch.count_nonzero(value.detach()).item() == 0)
        if key not in masked_keys:
            replaced[key] = value
            continue

        if key.split('_')[0] not in missing_embeddings:
            embed_shape = (1,) + tuple(value.shape[1:])
            missing_param = torch.nn.Parameter(
                torch.empty(embed_shape, device=value.device, dtype=value.dtype)
            )
            torch.nn.init.normal_(missing_param, mean=0.0, std=0.02)
            missing_embeddings[key.split('_')[0]] = missing_param
            if optimizer is not None:
                optimizer.add_param_group({'params': [missing_embeddings[key.split('_')[0]]]})

        missing_embed = missing_embeddings[key.split('_')[0]].to(device=value.device, dtype=value.dtype)
        expand_shape = (value.shape[0],) + tuple(missing_embed.shape[1:])
        replaced[key] = missing_embed.expand(expand_shape)

    return replaced


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


def _run_test_loop(
    model_to_test,
    encoder_to_test,
    test,
    modalities,
    args,
    setting,
    device,
    missing_embeddings,
    getattentionmap,
    header_label,
    log_prefix='test',
    wandb_extra=None,
    use_wandb=False,
    result_filename_prefix='result',
):
    """Run the test evaluation loop, append results to the per-config output file,
    and (optionally) log metrics to wandb. Returns rets (attention maps when
    ``getattentionmap`` is True, otherwise empty per-task lists)."""
    task_names = {'MOR': 'mortality', 'RAD': 'readmission'}
    model_to_test.eval()
    for enc in encoder_to_test.values():
        enc.eval()
    with torch.no_grad():
        rets = [[], [], [], []]
        test_log = {}
        print(f"\n{header_label}...")
        task_mods = mods_for_task(args)
        if args.transfer_moe:
            out_fname = f"{args.results_dir}/flame/multitask/{args.base_task}/mod_drop_rate_{args.modality_drop_rate}/{args.base_task_mods}/experts_{args.num_of_experts[0]}/{args.seed}/{result_filename_prefix}_{args.task}_{task_mods}_transfer_moe_from_{args.base_task}_mod_drop_rate_{args.modality_drop_rate}.txt"
        elif args.lora:
            out_fname = f"{args.results_dir}/{args.fusion_model}/{args.base_task}/mod_drop_rate_{args.modality_drop_rate}/{args.base_task_mods}/experts_{args.num_of_experts[0]}/{args.seed}/{result_filename_prefix}_{args.task}_{task_mods}_lora_from_{args.base_task}_mod_drop_rate_{args.modality_drop_rate}.txt"
        elif args.fine_tune:
            out_fname = f"{args.results_dir}/{args.fusion_model}/{args.base_task}/mod_drop_rate_{args.modality_drop_rate}/{args.base_task_mods}/experts_{args.num_of_experts[0]}/{args.seed}/{result_filename_prefix}_{args.task}_{task_mods}_ft_from_{args.base_task}_mod_drop_rate_{args.modality_drop_rate}.txt"
        elif args.linear_probe:
            out_fname = f"{args.results_dir}/{args.fusion_model}/{args.base_task}/mod_drop_rate_{args.modality_drop_rate}/{args.base_task_mods}/experts_{args.num_of_experts[0]}/{args.seed}/{result_filename_prefix}_{args.task}_{task_mods}_linear_probe_from_{args.base_task}_mod_drop_rate_{args.modality_drop_rate}.txt"
        elif args.cross_method == 'flexmoe':
            out_fname = f"{args.results_dir}/flexmoe/multitask/{args.task}/mod_drop_rate_{args.modality_drop_rate}/experts_{args.num_of_experts[0]}/{args.seed}/{result_filename_prefix}_{args.task}_lr{args.lr}_wd{args.weight_decay}_{task_mods}_mod_drop_rate_{args.modality_drop_rate}.txt"
        else:
            if args.shared_modality_encoders:
                if args.multitask_moe:
                    out_fname = f"{args.results_dir}/flame_w_balanced_loss_{args.balance_loss_coef}_alpha_{args.alpha}_w_residual_scaling/multitask/{args.gating_function[0]}/{args.task}/mod_drop_rate_{args.modality_drop_rate}/experts_{args.num_of_experts[0]}/{args.seed}/{result_filename_prefix}_{args.task}_{task_mods}_lr{args.lr}_wd{args.weight_decay}_mod_drop_rate_{args.modality_drop_rate}.txt"
                else:
                    out_fname = f"{args.results_dir}/{args.fusion_model}/multitask/{args.task}/mod_drop_rate_{args.modality_drop_rate}/experts_{args.num_of_experts[0]}/{args.seed}/{result_filename_prefix}_{args.task}_{task_mods}_lr{args.lr}_wd{args.weight_decay}_mod_drop_rate_{args.modality_drop_rate}.txt"
            else:
                out_fname = f"{args.results_dir}/{args.fusion_model}/{args.base_task}/mod_drop_rate_{args.modality_drop_rate}/{args.base_task_mods}/experts_{args.num_of_experts[0]}/{args.seed}/{result_filename_prefix}_{args.task}_{task_mods}_lr{args.lr}_wd{args.weight_decay}_mod_drop_rate_{args.modality_drop_rate}.txt"
        os.makedirs(os.path.dirname(out_fname), exist_ok=True)
        f = open(out_fname, 'a')
        f.write(f"\n################## {header_label} ##################\n")
        f.write(setting + "  \n")
        print(f"\nWriting results to {out_fname}\n")
        for ii in tqdm(range(len(test))):
            eval_vals = {}
            eval_logits = []
            eval_labels = []
            task = modalities[int(ii)][0].split('_')[1]

            if args.lora:
                model_to_test.base_model.model.model.to_logits = model_to_test.base_model.model.model.to_logitslist[ii]
            else:
                model_to_test.to_logits = model_to_test.to_logitslist[ii]
            for jj in tqdm(test[ii]):
                if task in ['IHM', 'PHENO', 'LOS']:
                    ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts, input_ids_sequences, attn_mask_sequences, text_emb, note_time, note_time_mask, cxr_feats, cxr_time, cxr_time_mask, ecg_feats, ecg_time, ecg_time_mask, label, cxr_missing, text_missing, ecg_missing = jj
                    embeddings = encoder_to_test[task](
                        x_ts=ts_input_sequences,
                        x_ts_mask=ts_mask_sequences,
                        ts_tt_list=ts_tt,
                        input_ids_sequences=input_ids_sequences,
                        attn_mask_sequences=attn_mask_sequences, text_emb=text_emb, note_time_list=note_time,
                        note_time_mask_list=note_time_mask,
                        cxr_feats=cxr_feats,
                        cxr_time=cxr_time,
                        cxr_time_mask=cxr_time_mask,
                        ecg_feats=ecg_feats,
                        ecg_time=ecg_time,
                        ecg_time_mask=ecg_time_mask, labels=label, reg_ts=reg_ts,
                        cxr_missing=cxr_missing, text_missing=text_missing, ecg_missing=ecg_missing, modalities=modalities[int(ii)]
                    )
                elif task in ['MOR', 'RAD']:
                    codes, types, timestamps, ages, genders, ethnicities, label = jj['codes'], jj['types'], jj['timestamps'], jj['age'], jj['gender'], jj['ethnicity'], jj[task_names[task]].long()
                    embeddings = encoder_to_test[task](
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
                    embeddings = encoder_to_test[task](
                        embed_cc=embed_cc, embed_mlo=embed_mlo, embed_2dcc=embed_2dcc, embed_2dmlo=embed_2dmlo, all_views=all_views, modalities=modalities[int(ii)], task=task
                    )
                elif task.lower() == 'diag':
                    _, label, mod_tensors = jj
                    embeddings = encoder_to_test[task](
                        mod_tensors=mod_tensors, modalities=modalities[int(ii)], task=task,
                    )
                indict = {}
                for i in range(0, len(modalities[ii])):
                    indict[modalities[ii][i]] = embeddings[modalities[ii][i]].float().to(device)
                indict, masked_keys = drop_modalities(indict, args.modality_drop_rate)
                if args.modality_drop_rate > 0:
                    indict = replace_missing_embeddings(indict, missing_embeddings, masked_keys=masked_keys)

                out, balance_loss = model_to_test(indict=indict, task=task) if args.lora else model_to_test(indict, task=task)
                if 'PHENO' in modalities[int(ii)][0]:
                    logit = torch.nn.functional.sigmoid(out)
                elif 'birads' in modalities[int(ii)][0].lower() or 'density' in modalities[int(ii)][0].lower() or 'diag' in modalities[int(ii)][0].lower():
                    logit = torch.nn.functional.softmax(out, dim=-1)
                else:
                    logit = torch.nn.functional.softmax(out, dim=-1)[:, 1]
                logits = logit.cpu().numpy()
                labels = label.cpu().numpy()
                eval_logits += logits.tolist()
                eval_labels += labels.tolist()
                if getattentionmap:
                    rets[ii].append(model_to_test.attns)
            all_logits = np.array(eval_logits)
            all_label = np.array(eval_labels)

            if 'PHENO' in modalities[int(ii)][0]:
                all_pred = np.where(all_logits > 0.5, 1, 0)
                eval_vals = metrics_multilabel(all_label, all_logits, verbose=0)
                eval_vals['micro_f1'] = f1_score(all_label, all_pred, average='micro')
                eval_vals['macro_f1'] = f1_score(all_label, all_pred, average='macro')
                eval_vals['weighted_f1'] = f1_score(all_label, all_pred, average='weighted')
                eval_vals['subset_accuracy'] = accuracy_score(all_label, all_pred)
                eval_vals['hamming_accuracy'] = 1.0 - hamming_loss(all_label, all_pred)
                test_log[f'{log_prefix}/{task}/auc_mean'] = float(eval_vals['auc_scores'].mean())
                test_log[f'{log_prefix}/{task}/auprc_mean'] = float(np.asarray(eval_vals['auprc_scores']).mean())
            elif 'birads' in modalities[int(ii)][0].lower() or 'density' in modalities[int(ii)][0].lower() or 'diag' in modalities[int(ii)][0].lower():
                eval_vals = metrics_multiclass(all_label, all_logits, verbose=0)
                all_pred = np.argmax(all_logits, axis=1)
                print("label dist:", np.bincount(all_label.astype(int)))
                print("pred dist :", np.bincount(all_pred.astype(int)))
                eval_vals['micro_f1'] = f1_score(all_label, all_pred, average='micro')
                eval_vals['macro_f1'] = f1_score(all_label, all_pred, average='macro')
                eval_vals['weighted_f1'] = f1_score(all_label, all_pred, average='weighted')
                eval_vals['accuracy'] = accuracy_score(all_label, all_pred)
                test_log[f'{log_prefix}/{task}/ave_auc_macro'] = float(eval_vals['ave_auc_macro'])
                if eval_vals.get('ave_auprc_macro') is not None:
                    test_log[f'{log_prefix}/{task}/ave_auprc_macro'] = float(eval_vals['ave_auprc_macro'])
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
                test_log[f'{log_prefix}/{task}/auc'] = float(eval_vals['auc'])

            for metric_name, metric_val in eval_vals.items():
                scalar_val = _to_float_if_scalar(metric_val)
                if scalar_val is not None:
                    test_log[f'{log_prefix}/{task}/{metric_name}'] = scalar_val

            f.write(f"------Task {ii}------\n")
            for k, v in eval_vals.items():
                f.write(k + ': {}'.format(v))
                f.write('\n')
                f.write('\n')
        f.close()
        if use_wandb:
            log_dict = {**test_log}
            if wandb_extra:
                log_dict.update(wandb_extra)
            wandb.log(log_dict)
    return rets


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
    if args.lora or args.transfer_moe:
        # Only trainable parameters (frozen params excluded)
        params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))
        for enc in encoder.values():
            params_to_optimize += list(filter(lambda p: p.requires_grad, enc.parameters()))
        optim = optimizer(params_to_optimize, lr=lr, weight_decay=weight_decay)
    else:
        # For full fine-tuning: all model parameters + all encoder parameters
        # params_to_optimize = list(model.parameters())
        # for enc in encoder.values():
        #     params_to_optimize += list(enc.parameters())
        # optim = optimizer(params_to_optimize, lr=lr, weight_decay=weight_decay)

        moe_params = [p for n, p in model.named_parameters()
                    if 'experts' in n.lower() or 'router' in n.lower() or 'w_gate' in n.lower() or 'w_noise' in n.lower()]
        other_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and id(p) not in {id(x) for x in moe_params}]
        enc_params = [p for enc in encoder.values() for p in enc.parameters() if p.requires_grad]

        optim = torch.optim.AdamW([
            {'params': other_params + enc_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': moe_params, 'lr': lr, 'weight_decay': weight_decay},   # 5× LR, NO weight decay
        ])


    missing_embeddings = torch.nn.ParameterDict()
    testaccs = []
    rets = [[], [], [], []]
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

    # --- MoE diagnostics: log file lives next to the checkpoint (savedir) ---
    moe_diag_dir = os.path.dirname(savedir) if savedir and os.path.splitext(savedir)[1] else savedir
    if not moe_diag_dir:
        moe_diag_dir = "."
    moe_diag_tag = os.path.splitext(os.path.basename(savedir))[0] if savedir else "run"

    use_wandb = bool(getattr(args, 'use_wandb', False) or getattr(args, 'wandb', False))
    wandb_run_started_here = False
    if use_wandb and wandb is None:
        print("[warn] wandb logging requested but wandb is not installed. Continuing without wandb.")
        use_wandb = False
    if use_wandb and wandb.run is None:
        # Map short task names to their mod arg names (e.g. mortality -> mor_mod)
        _task_to_mod_arg = {
            'ihm': 'ihm_mod', 'los': 'los_mod', 'pheno': 'pheno_mod',
            'mortality': 'mor_mod', 'readmission': 'rad_mod',
            'birads': 'birads_mod', 'risk': 'risk_mod', 'density': 'density_mod',
            'diag': 'diag_mod',
        }
        mods_str = "_".join([
            getattr(args, _task_to_mod_arg.get(t, f"{t}_mod"), "?")
            for t in args.task.split("-")
        ])
        default_run_name = (
            f"{args.fusion_model}_{args.task}_{mods_str}"
            f"_balance_coeff_{args.balance_loss_coef}"
            f"_num_experts_experts_{args.num_of_experts[0]}_multitask_run"
        )
        _wandb_entity = getattr(args, 'wandb_entity', None) or os.environ.get('WANDB_ENTITY')
        _wandb_kwargs = dict(
            project=getattr(args, 'wandb_project', 'clinical-highmmt'),
            name=getattr(args, 'wandb_run_name', None) or default_run_name,
            config=vars(args) if hasattr(args, '__dict__') else None,
        )
        if _wandb_entity:
            _wandb_kwargs['entity'] = _wandb_entity
        wandb.init(**_wandb_kwargs)
        wandb_run_started_here = True

    _wandb_for_loggers = wandb if use_wandb else None
    moe_diag = MoEDiagnosticsLogger(
        log_dir=moe_diag_dir,
        jsonl_name=f"moe_diag_{moe_diag_tag}_lr{args.lr}_wd{args.weight_decay}.jsonl",
        text_name=f"moe_diag_{moe_diag_tag}_lr{args.lr}_wd{args.weight_decay}.txt",
        wandb_run=_wandb_for_loggers,
    )
    moe_diag.register_hooks(model)
    layerwise_grad_logger = LayerwiseGradLogger(
        log_dir=moe_diag_dir,
        model_jsonl=f"layerwise_grads_model_{moe_diag_tag}_lr{args.lr}_wd{args.weight_decay}.jsonl",
        encoder_jsonl=f"layerwise_grads_encoder_{moe_diag_tag}_lr{args.lr}_wd{args.weight_decay}.jsonl",
        wandb_run=_wandb_for_loggers,
    )

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
                elif task.lower() == 'diag':
                    _, label, mod_tensors = js[ii]
                    embeddings = encoder[task](
                        mod_tensors=mod_tensors, modalities=modalities[int(ii)], task=task,
                    )

                if args.lora:
                    model.base_model.model.model.to_logits = model.base_model.model.model.to_logitslist[int(ii)]
                else:
                    model.to_logits = model.to_logitslist[int(ii)]
                indict={}
                for i in range(len(modalities[int(ii)])):
                    indict[modalities[int(ii)][i]] = embeddings[modalities[int(ii)][i]].float().to(device)
                
                indict, masked_keys = drop_modalities(indict, args.modality_drop_rate)
                if args.modality_drop_rate > 0:
                    indict = replace_missing_embeddings(indict, missing_embeddings, masked_keys=masked_keys, optimizer=optim)
                
                if recon:
                    out, rec, balance_loss = model(indict=indict, task=task, use_recon=True) if args.lora else model(indict, task=task, use_recon=True)
                    stuffs = []
                    for modal in indict:
                        stuffs.append(torch.mean(indict[modal], dim=1))
                    origs = torch.cat(stuffs, dim=1)
                    loss = criterion[int(ii)](out, label.to(device)) + recon_weight * recon_criterion(rec, origs) + args.balance_loss_coef * balance_loss
                else:
                    out, balance_loss = model(indict=indict, task=task) if args.lora else model(indict, task=task)
                    if 'PHENO' in modalities[int(ii)][0]:
                        loss=criterion[int(ii)](out, label.float().to(device))
                    elif 'birads' in modalities[int(ii)][0].lower() or 'density' in modalities[int(ii)][0].lower() or 'diag' in modalities[int(ii)][0].lower():
                        loss=criterion[int(ii)](out, label.to(device))
                    else:
                        loss=criterion[int(ii)](out, label.to(device))
                    if balance_loss is not None:
                        loss = loss + args.balance_loss_coef * balance_loss
                losses += loss * train_weights[int(ii)]
            losses.backward()
            batch_model_grad_norm = _grad_l2_norm(model_grad_params)
            batch_encoder_grad_norm = _grad_l2_norm(encoder_grad_params)
            # Log MoE-vs-encoder grad norms once per epoch (first batch)
            moe_diag.log_grad_norms(model, ep)
            layerwise_grad_logger.log(model, encoder, ep)
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

        # --- MoE diagnostics: epoch-level metrics from hooks on the most recent forward ---
        moe_diag.log_epoch(model, ep)

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
                    elif task.lower() == 'diag':
                        _, label, mod_tensors = jj
                        embeddings = encoder[task](
                            mod_tensors=mod_tensors, modalities=modalities[int(ii)], task=task,
                        )
                    if args.lora:
                        model.base_model.model.model.to_logits = model.base_model.model.model.to_logitslist[ii]
                    else:
                        model.to_logits = model.to_logitslist[ii]
                    indict={}
                    for i in range(len(modalities[ii])):
                        indict[modalities[ii][i]] = embeddings[modalities[ii][i]].float().to(device)
                    indict, masked_keys = drop_modalities(indict, args.modality_drop_rate)
                    if args.modality_drop_rate > 0:
                        indict = replace_missing_embeddings(indict, missing_embeddings, masked_keys=masked_keys)
                    
                    if recon:
                        out, rec, balance_loss = model(indict=indict, task=task, use_recon=True) if args.lora else model(indict, task=task, use_recon=True)
                        stuffs = []
                        for modal in indict:
                            stuffs.append(torch.mean(indict[modal], dim=1))
                        origs = torch.cat(stuffs, dim=1)
                        val_loss = criterion[int(ii)](out, label.to(device)) + recon_weight * recon_criterion(rec, origs)
                    else:
                        out, balance_loss = model(indict=indict, task=task) if args.lora else model(indict, task=task)
                        if 'PHENO' in modalities[int(ii)][0]:
                            val_loss = criterion[int(ii)](out, label.float().to(device))
                        elif 'birads' in modalities[int(ii)][0].lower() or 'density' in modalities[int(ii)][0].lower() or 'diag' in modalities[int(ii)][0].lower():
                            val_loss = criterion[int(ii)](out, label.to(device))
                        else:
                            val_loss = criterion[int(ii)](out, label.to(device))
                    if balance_loss is not None:
                        val_loss = val_loss + args.balance_loss_coef * balance_loss
                    val_loss_task_sum += val_loss.item()
                    val_loss_task_steps += 1
                    val_loss_total_sum += val_loss.item()
                    val_loss_total_steps += 1

                    if 'PHENO' in modalities[int(ii)][0]:
                        logit = torch.nn.functional.sigmoid(out)
                    elif 'birads' in modalities[int(ii)][0].lower() or 'density' in modalities[int(ii)][0].lower() or 'diag' in modalities[int(ii)][0].lower():
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
                elif 'birads' in modalities[int(ii)][0].lower() or 'density' in modalities[int(ii)][0].lower() or 'diag' in modalities[int(ii)][0].lower():
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
                # Hooks contain closures that cannot be pickled; drop them before save.
                moe_diag.remove_hooks()
                try:
                    torch.save(model, savedir)
                    for ii in range(len(modalities)):
                        task = modalities[int(ii)][0].split('_')[1]
                        torch.save(encoder[task], f'{savedir.split(".pt")[0]}_{task}_mod_drop_rate_{args.modality_drop_rate}_encoder.pt')
                finally:
                    moe_diag.register_hooks(model)
                val_log['val/best_score_sum'] = float(bestacc)
        print('Model saved to ', savedir)
        if use_wandb:
            wandb.log({**train_log, **val_log, 'epoch': ep})
        # import pdb; pdb.set_trace()
        ### Testing function ###
        if (ep+1)%10==0 or ep==args.num_train_epochs-1:
            rets = _run_test_loop(
                model, encoder, test, modalities, args, setting, device,
                missing_embeddings, getattentionmap,
                header_label=f"Epoch {ep}",
                log_prefix='test',
                wandb_extra={'epoch': ep},
                use_wandb=use_wandb,
            )

    # --- Final test run on best saved model ---
    if args.num_train_epochs > 0 and savedir and os.path.exists(savedir):
        print('\nLoading best model from checkpoint for final test run...')
        moe_diag.remove_hooks()
        best_model = torch.load(savedir, map_location=device).to(device)
        best_encoder = {}
        for ii in range(len(modalities)):
            task = modalities[int(ii)][0].split('_')[1]
            enc_path = f'{savedir.split(".pt")[0]}_{task}_mod_drop_rate_{args.modality_drop_rate}_encoder.pt'
            best_encoder[task] = torch.load(enc_path, map_location=device).to(device)
        rets = _run_test_loop(
            best_model, best_encoder, test, modalities, args, setting, device,
            missing_embeddings, getattentionmap,
            header_label="Final Best Model Test",
            log_prefix='best_test',
            wandb_extra=None,
            use_wandb=use_wandb,
            result_filename_prefix='best_model_results',
        )

    if use_wandb and wandb_run_started_here:
        wandb.finish()
    moe_diag.close()
    if getattentionmap:
        return rets
    return testaccs
