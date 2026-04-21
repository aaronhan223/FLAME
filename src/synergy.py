import sys
import os
import argparse

sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# FIX: Added missing import to resolve NameError
from transformers import AutoTokenizer 

# Internal project imports
from src.mimiciv_task_setup import setup_tasks_and_modalities
from src.utils import set_seed

# 1. PID Estimator: Shallow MLP to learn unnormalized joint distribution A 
class SynergyEstimator(nn.Module):
    def __init__(self, feature_dim, num_labels):
        super().__init__()
        self.f1 = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, 64))
        self.f2 = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, 64))
        self.label_embed = nn.Embedding(num_labels, 64)

    def forward(self, x1, x2, y):
        # learners of an outer-product similarity matrix A
        feat1 = self.f1(x1) * self.label_embed(y)
        feat2 = self.f2(x2) * self.label_embed(y)
        A = torch.matmul(feat1, feat2.t()) 
        return torch.exp(A)

# 2. Sinkhorn-Knopp: Iterative row/column normalization to obtain a
#    doubly-stochastic matrix from the unnormalized affinity matrix A.
def sinkhorn_knopp(A, iterations=20):
    for _ in range(iterations):
        # Row normalization
        A = A / (A.sum(dim=1, keepdim=True) + 1e-9)
        # Column normalization
        A = A / (A.sum(dim=0, keepdim=True) + 1e-9)
    return A

def main():
    parser = argparse.ArgumentParser()
    # (All existing arguments preserved)
    parser.add_argument("--encoder_a_path", type=str, required=True)
    parser.add_argument("--encoder_b_path", type=str, required=True)
    parser.add_argument("--task", type=str, default='readmission-density')
    parser.add_argument("--mimic_path", type=str)
    parser.add_argument("--eicu_path", type=str)
    parser.add_argument("--embed_path", type=str)
    parser.add_argument("--ihm_mod", type=str, default='')
    parser.add_argument("--los_mod", type=str, default='')
    parser.add_argument("--pheno_mod", type=str, default='')
    parser.add_argument("--rad_mod", type=str, default='')
    parser.add_argument("--mor_mod", type=str, default='')
    parser.add_argument("--birads_mod", type=str, default='')
    parser.add_argument("--risk_mod", type=str, default='')
    parser.add_argument("--density_mod", type=str, default='')
    parser.add_argument("--tensorboard_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--eval_score", default=['auc', 'auprc', 'f1'], type=list)
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--train_bs_mimic", type=int, default=8)
    parser.add_argument("--train_bs_eicu", type=int, default=8)
    parser.add_argument("--train_bs_embed", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_update_bert_epochs", type=int, default=10)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--txt_learning_rate", type=float, default=5e-5)
    parser.add_argument("--ts_learning_rate", type=float, default=0.0004)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--pt_mask_ratio", default=0.15, type=float)
    parser.add_argument("--mean_mask_length", default=3, type=int)
    parser.add_argument('--chunk', action='store_true')
    parser.add_argument("--chunk_type", default='sent_doc_pos', type=str)
    parser.add_argument("--warmup_proportion", default=0.10, type=float)
    parser.add_argument("--kernel_size", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--cross_layers", type=int, default=3)
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--perceiver_dim", default=64, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--irregular_learn_emb_ts", action='store_true')
    parser.add_argument("--irregular_learn_emb_text", action='store_true')
    parser.add_argument("--irregular_learn_emb_cxr", action='store_true')
    parser.add_argument("--irregular_learn_emb_ecg", action='store_true')
    parser.add_argument("--reg_ts", action='store_true')
    parser.add_argument("--tt_max", default=48, type=int)
    parser.add_argument("--tt_max_eicu", default=1, type=int)
    parser.add_argument("--embed_time", default=64, type=int)
    parser.add_argument('--ts_to_txt', action='store_true')
    parser.add_argument('--txt_to_ts', action='store_true')
    parser.add_argument("--dropout", default=0.10, type=float)
    parser.add_argument("--model_name", default='BioBert', type=str)
    parser.add_argument('--num_of_notes', type=int, default=5)
    parser.add_argument('--notes_order', default=None)
    parser.add_argument('--ratio_notes_order', type=float, default=None)
    parser.add_argument('--bertcount', type=int, default=3)
    parser.add_argument('--first_n_item', type=int, default=3)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--self_cross', action='store_true')
    parser.add_argument('--TS_mixup', action='store_true')
    parser.add_argument('--mixup_level', type=str, default='batch')
    parser.add_argument('--cross_method', type=str, default='moe')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--generate_data', action='store_true')
    parser.add_argument('--FTLSTM', action='store_true')
    parser.add_argument('--Interp', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument("--datagereate_seed", type=int, default=42)
    parser.add_argument("--TS_model", type=str, default='Atten')
    parser.add_argument("--use_pt_text_embeddings", action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--base_task_mods', type=str, default='')
    parser.add_argument('--base_task', type=str, default='')
    parser.add_argument('--results_dir', type=str, default='./results') 
    parser.add_argument('--fusion_model', type=str, default='multimodalityperceiver')
    parser.add_argument('--linear_probe', action='store_true')
    parser.add_argument('--shared_modality_encoders', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='clinical-highmmt')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument("--num_of_experts", nargs='*', type=int)
    parser.add_argument("--top_k", nargs='*', type=int)
    parser.add_argument("--router_type", default='joint', type=str)
    parser.add_argument("--gating_function", nargs='*', type=str)
    parser.add_argument("--modality_drop_rate", default=0.0, type=float)
    parser.add_argument("--multitask_moe", action='store_true')

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    task_mapping = {'ihm': args.ihm_mod, 'los': args.los_mod, 'pheno': args.pheno_mod, 
                    'readmission': args.rad_mod, 'mortality': args.mor_mod, 
                    'birads': args.birads_mod, 'risk': args.risk_mod, 'density': args.density_mod}
    modeltype = {t: '_'.join(sorted(task_mapping[t].split("-"))) for t in args.task.split("-") if t in task_mapping and task_mapping[t]}

    _, all_valid, _, _, modalities_per_task, _, _, _, _ = setup_tasks_and_modalities(
        args=args, device=device, tokenizer=tokenizer, modeltype=modeltype, modalities=set(), BioBert=None
    )

    # Expect exactly two tasks, e.g. "ihm-risk", "readmission-birads", etc.
    task_names = args.task.split("-")
    if len(task_names) != 2:
        raise ValueError(f"Expected exactly two tasks in --task, got {args.task}")
    task_a_name, task_b_name = task_names[0], task_names[1]

    enc_a = torch.load(args.encoder_a_path, map_location=device).eval()
    enc_b = torch.load(args.encoder_b_path, map_location=device).eval()

    estimator = SynergyEstimator(feature_dim=args.embed_dim, num_labels=args.num_labels).to(device)
    optimizer = optim.Adam(estimator.parameters(), lr=1e-4)

    # Accumulators
    total_ip, total_iq, valid_batches = 0.0, 0.0, 0

    print(f"Calculating Synergy via BATCH Algorithm for tasks: {task_a_name} vs {task_b_name} ...")
    for epoch in range(args.num_train_epochs):
        for batch_a, batch_b in tqdm(zip(all_valid[0], all_valid[1]), total=min(len(all_valid[0]), len(all_valid[1]))):
            with torch.no_grad():
                # ---------- Task A Extraction ----------
                if task_a_name in ["ihm", "los", "pheno"]:
                    # MIMIC-IV time-series + text + CXR (+/- ECG)
                    ts_input, ts_mask, ts_tt, reg_ts, input_ids, attn_mask, text_emb, \
                    note_time, note_time_mask, cxr_feats, cxr_time, cxr_time_mask, \
                    ecg_feats, ecg_time, ecg_time_mask, label_a, cxr_missing, \
                    text_missing, ecg_missing = batch_a

                    feat_a = enc_a(
                        x_ts=ts_input.to(device), x_ts_mask=ts_mask.to(device),
                        ts_tt_list=ts_tt, reg_ts=reg_ts.to(device),
                        input_ids_sequences=input_ids.to(device),
                        attn_mask_sequences=attn_mask.to(device),
                        text_emb=text_emb, note_time_list=note_time,
                        note_time_mask_list=note_time_mask, cxr_feats=cxr_feats,
                        cxr_time=cxr_time, cxr_time_mask=cxr_time_mask,
                        ecg_feats=ecg_feats, ecg_time=ecg_time,
                        ecg_time_mask=ecg_time_mask, labels=label_a,
                        cxr_missing=cxr_missing, text_missing=text_missing,
                        ecg_missing=ecg_missing, modalities=modalities_per_task[0]
                    )
                elif task_a_name in ["readmission", "mortality"]:
                    # eICU EHR sequence
                    codes = batch_a["codes"]
                    types = batch_a["types"]
                    timestamps = batch_a["timestamps"]
                    ages = batch_a["age"]
                    genders = batch_a["gender"]
                    ethnicities = batch_a["ethnicity"]
                    if task_a_name == "readmission":
                        label_a = batch_a["readmission"].long()
                    else:
                        label_a = batch_a["mortality"].long()

                    feat_a = enc_a(
                        codes=codes,
                        types=types,
                        timestamps=timestamps,
                        ages=ages,
                        genders=genders,
                        ethnicities=ethnicities,
                        modalities=modalities_per_task[0],
                    )
                elif task_a_name in ["birads", "risk", "density"]:
                    # EMBED mammography encodings as Task A
                    idx_a, label_a, embed_2dcc_a, embed_2dmlo_a, embed_cc_a, embed_mlo_a, all_views_a = batch_a
                    all_views_input_a = all_views_a.to(device) if all_views_a is not None else None
                    # Derive EMBED task code (e.g. 'RISK', 'BIRADS', 'DENSITY')
                    task_a_code = modalities_per_task[0][0].split('_')[1]
                    feat_a = enc_a(
                        embed_cc=embed_cc_a.to(device), embed_mlo=embed_mlo_a.to(device),
                        embed_2dcc=embed_2dcc_a.to(device), embed_2dmlo=embed_2dmlo_a.to(device),
                        all_views=all_views_input_a, modalities=modalities_per_task[0],
                        task=task_a_code,
                    )
                else:
                    raise ValueError(f"Unsupported task A name: {task_a_name}")

                # ---------- Task B Extraction ----------
                if task_b_name in ["ihm", "los", "pheno"]:
                    ts_input_b, ts_mask_b, ts_tt_b, reg_ts_b, input_ids_b, attn_mask_b, text_emb_b, \
                    note_time_b, note_time_mask_b, cxr_feats_b, cxr_time_b, cxr_time_mask_b, \
                    ecg_feats_b, ecg_time_b, ecg_time_mask_b, label_b, cxr_missing_b, \
                    text_missing_b, ecg_missing_b = batch_b

                    feat_b = enc_b(
                        x_ts=ts_input_b.to(device), x_ts_mask=ts_mask_b.to(device),
                        ts_tt_list=ts_tt_b, reg_ts=reg_ts_b.to(device),
                        input_ids_sequences=input_ids_b.to(device),
                        attn_mask_sequences=attn_mask_b.to(device),
                        text_emb=text_emb_b, note_time_list=note_time_b,
                        note_time_mask_list=note_time_mask_b, cxr_feats=cxr_feats_b,
                        cxr_time=cxr_time_b, cxr_time_mask=cxr_time_mask_b,
                        ecg_feats=ecg_feats_b, ecg_time=ecg_time_b,
                        ecg_time_mask=ecg_time_mask_b, labels=label_b,
                        cxr_missing=cxr_missing_b, text_missing=text_missing_b,
                        ecg_missing=ecg_missing_b, modalities=modalities_per_task[1],
                    )
                elif task_b_name in ["readmission", "mortality"]:
                    codes_b = batch_b["codes"]
                    types_b = batch_b["types"]
                    timestamps_b = batch_b["timestamps"]
                    ages_b = batch_b["age"]
                    genders_b = batch_b["gender"]
                    ethnicities_b = batch_b["ethnicity"]
                    if task_b_name == "readmission":
                        label_b = batch_b["readmission"].long()
                    else:
                        label_b = batch_b["mortality"].long()

                    feat_b = enc_b(
                        codes=codes_b,
                        types=types_b,
                        timestamps=timestamps_b,
                        ages=ages_b,
                        genders=genders_b,
                        ethnicities=ethnicities_b,
                        modalities=modalities_per_task[1],
                    )
                elif task_b_name in ["birads", "risk", "density"]:
                    idx_b, label_b, embed_2dcc_b, embed_2dmlo_b, embed_cc_b, embed_mlo_b, all_views_b = batch_b
                    all_views_input_b = all_views_b.to(device) if all_views_b is not None else None
                    task_b_code = modalities_per_task[1][0].split('_')[1]
                    feat_b = enc_b(
                        embed_cc=embed_cc_b.to(device), embed_mlo=embed_mlo_b.to(device),
                        embed_2dcc=embed_2dcc_b.to(device), embed_2dmlo=embed_2dmlo_b.to(device),
                        all_views=all_views_input_b, modalities=modalities_per_task[1],
                        task=task_b_code,
                    )
                else:
                    raise ValueError(f"Unsupported task B name: {task_b_name}")

                # Skip batches where one of the encoders produced no modality embeddings.
                if (not feat_a) or (not feat_b):
                    print(f"[skip] Empty embeddings. Task A keys: {list(feat_a.keys())}, Task B keys: {list(feat_b.keys())}")
                    continue

                # Aggregate over time, then average across modalities so that
                # both tasks have a fixed embedding size (args.embed_dim),
                # independent of how many modalities each task uses.
                reps_a = [feat_a[m].mean(dim=1) for m in feat_a]  # list of [B, D]
                reps_b = [feat_b[m].mean(dim=1) for m in feat_b]  # list of [B, D]

                rep_a = torch.stack(reps_a, dim=0).mean(dim=0)    # [B_a, D]
                rep_b = torch.stack(reps_b, dim=0).mean(dim=0)    # [B_b, D]

                # Align batch sizes across the two tasks by truncating
                # to the smaller batch size so that elementwise operations
                # (e.g., rep_a + rep_b) are well-defined.
                batch_size = min(rep_a.size(0), rep_b.size(0))
                rep_a = rep_a[:batch_size]
                rep_b = rep_b[:batch_size]
                label_a = label_a[:batch_size]

                p_y_full = F.softmax(rep_a + rep_b, dim=-1)
                p_y_marginal = p_y_full.mean(dim=0)
                batch_ip = torch.mean(torch.log(p_y_full / (p_y_marginal + 1e-9)))

            # 4. BATCH Optimization
            A_unnorm = estimator(rep_a, rep_b, label_a.to(device))
            q_star = sinkhorn_knopp(A_unnorm)
            # Use the normalized joint distribution q_star to obtain an
            # alternative information-like quantity. Since q_star is
            # doubly-stochastic over the batch indices, we simply
            # aggregate its log-probabilities.
            batch_iq = torch.mean(torch.log(q_star + 1e-9))
            
            optimizer.zero_grad()
            batch_iq.backward()
            optimizer.step()

            total_ip += batch_ip.item()
            total_iq += batch_iq.item()
            valid_batches += 1

    if valid_batches > 0:
        print(f"\nOptimization Complete. Synergy: {(total_ip/valid_batches) - (total_iq/valid_batches):.6f}")
    else:
        print("\nError: No valid batches were processed. Check modality and task mapping.")

if __name__ == "__main__":
    main()