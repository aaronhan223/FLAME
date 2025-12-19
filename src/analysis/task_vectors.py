import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pandas as pd
import src.crossattnperceiver as crossattnperceiver
import sys
sys.modules['crossattnperceiver'] = crossattnperceiver
from src.crossattnperceiver import MultiModalityPerceiver, InputModality, PerceiverWrapper
from src.encoders import ModalityEncoders, FSEncoder
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib.pyplot as plt
import numpy as np
from src.analysis.evaluation import evaluate_model
import argparse
from transformers import set_seed
from src.analysis.utils import *
import src.get_data_eicu as get_data_eicu
sys.modules['get_data_eicu'] = get_data_eicu
# from src.get_data_eicu import data_prepare as eicu_data_prepare

def parse_args():
    parser = argparse.ArgumentParser(description="Alignment text and ts data")
    parser.add_argument(
            "--task", type=str, default="ihm-los-pheno"
        )
    parser.add_argument(
        "--mimic_path", type=str, default="/cis/home/xhan56/code/Multimodal-Transformer/src/Data/ihm", help="A path to dataset folder"
    )
    parser.add_argument(
        "--eicu_path", type=str, default="/cis/home/xhan56/code/clinical-highmmt/src/datasets/eicu/processed", help="A path to dataset folder"
    )
    parser.add_argument("--ihm_mod", type=str, default='', help="Modality compoenents for IHM task.")
    parser.add_argument("--los_mod", type=str, default='', help="Modality compoenents for LOS task.")
    parser.add_argument("--pheno_mod", type=str, default='', help="Modality compoenents for PHENO task.")
    parser.add_argument("--rad_mod", type=str, default='', help="Modality compoenents for readmission task.")
    parser.add_argument("--mor_mod", type=str, default='', help="Modality compoenents for mortality task.")

    parser.add_argument("--tensorboard_dir", type=str, default=None, help="Where to store the final model.")

    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--mode", type=str, default="train", help="train/test")
    parser.add_argument("--eval_score", default=['auc', 'auprc', 'f1'], type=list)

    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128, help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated," " sequences shorter will be padded if `--pad_to_max_lengh` is passed."),)
    parser.add_argument( "--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.", )
    parser.add_argument( "--model_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_bs_mimic",
        type=int,
        default=8,
        help="Batch size for the mimic training dataloader.",
    )
    parser.add_argument(
        "--train_bs_eicu",
        type=int,
        default=8,
        help="Batch size for the eicu training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument("--num_update_bert_epochs", type=int, default=10, help="Number of per training epochs update the bert model.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")

    parser.add_argument(
        "--txt_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate for Txt self-attention and Bert to use.",
    )

    parser.add_argument(
        "--ts_learning_rate",
        type=float,
        default=0.0004,
        help="Initial learning rate for TS self-attention to use.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument( "--pt_mask_ratio",default=0.15, type=float, help="mask rate for pretrain .",
    )
    parser.add_argument( "--mean_mask_length",default=3, type=int, help="mean mask length for pretrain .",
    )

    parser.add_argument('--chunk', action='store_true')
    parser.add_argument("--chunk_type", default='sent_doc_pos', type=str, help="How to chunk the text. sent_doc_pos: sentence level position + doc level position")
    parser.add_argument("--warmup_proportion", default=0.10, type=float, help="proportion for the warmup in the lr scheduler.")
    parser.add_argument("--kernel_size", type=int, default=1, help="Kernel size for CNN.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads.")
    parser.add_argument("--layers", type=int, default=3, help="Number of transformer encoder layer.")
    parser.add_argument("--cross_layers", type=int, default=3, help="Number of transformer cross encoder layer.")
    parser.add_argument("--embed_dim", default=30, type=int, help="attention embedding dim.")
    parser.add_argument("--perceiver_dim", default=64, type=int, help="perceiver latent dimension.")
    parser.add_argument("--hidden_size", default=128, type=int, help="linear layer hidden unit size.")

    parser.add_argument("--irregular_learn_emb_ts", action='store_true')
    parser.add_argument("--irregular_learn_emb_text", action='store_true')
    parser.add_argument("--irregular_learn_emb_cxr", action='store_true')
    parser.add_argument("--irregular_learn_emb_ecg", action='store_true')
    parser.add_argument("--reg_ts", action='store_true')
    parser.add_argument("--tt_max", default=48, type=int, help="max time for irregular time series.")
    parser.add_argument("--embed_time", default=64, type=int, help="emdedding for time.")
    parser.add_argument('--ts_to_txt', action='store_true')
    parser.add_argument('--txt_to_ts', action='store_true')

    parser.add_argument("--dropout", default=0.10, type=float, help="dropout.")
    parser.add_argument("--model_name", default='BioBert', type=str, help="model for text")
    parser.add_argument('--num_of_notes', help='Number of notes to include for a patient input 0 for all the notes', type=int, default=5)
    parser.add_argument('--notes_order', help='Should we get notes from beginning of the admission time or from end of it, options are: 1. First: pick first notes 2. Last: pick last notes', default=None)
    parser.add_argument('--ratio_notes_order', help='The parameter of a bernulli distribution on whether take notes from First or Last, 1-Last, 0-First',type=float, default=None)

    parser.add_argument('--bertcount',type=int, default=3,help='number of count update bert in total')
    parser.add_argument('--first_n_item', help='Top n item in val seeds', type=int, default=3)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--self_cross', action='store_true')
    parser.add_argument('--TS_mixup', action='store_true', help='mix up reg and irg data')

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--generate_data', action='store_true')
    parser.add_argument('--FTLSTM', action='store_true')
    parser.add_argument('--Interp', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument("--datagereate_seed", type=int, default=42, help="A seed for reproducible data generation .")
    parser.add_argument("--TS_model", type=str, default='Atten', help="LSTM, CNN, Atten")
    parser.add_argument("--use_pt_text_embeddings", action='store_true', help="Option to use pre-extracted text embeddings")
    parser.add_argument('--lora', action='store_true', help='Use LoRA for fine-tuning')
    parser.add_argument('--base_task_mods', type=str, default='', help='Modalities used in the base task for transfer learning')
    parser.add_argument('--base_task', type=str, default='', help='Base task for transfer learning')
    parser.add_argument('--new_task_mods', type=str, default='', help='Modalities used in the new task for transfer learning')
    parser.add_argument('--results_dir', type=str, default='/cis/home/schaud35/clinical-highmmt/src/analysis/', help='Directory to store results') 
    parser.add_argument('--fusion_model', type=str, default='multimodalityperceiver', help='Fusion model to use, Perceiver or CrossAttnTransformer')
    parser.add_argument('--log', type=bool, default=True)
    args = parser.parse_args()
    return args

def get_task_vectors(base_model, model_ft):
    # Compute task vector (parameter differences)
    task_vector = {}

    for (name, base_param), (name_ft, finetuned_param) in zip(base_model.state_dict().items(), model_ft.state_dict().items()):
        if name != name_ft or base_param.shape != finetuned_param.shape:
            continue
        task_vector[name] = finetuned_param - base_param
    
    return task_vector

def print_layerwise_ranks(model, tol=1e-4, log=False):
    ranks = layerwise_rank_analysis(model, tol=tol)
    if log:
        print("\n\nLayerwise ranks for model:")
        for id, (name, info) in enumerate(ranks.items()):
            print(f"[{id}] {name}: rank = {info['rank']} / {min(info['shape'])} ({info['rank_ratio']:.2f})")
    return ranks

def print_layerwise_concat_ranks(model1, model2, tol=1e-4, log=False): 
    ranks = layerwise_concat_rank(model1, model2, tol=tol)
    if log:
        print("\n\nConcatenated layerwise ranks:")
        for id, (name, info) in enumerate(ranks.items()):
            print(f"[{id}] {name}: concat rank = {info['concat_rank']} ({info['concat_rank_ratio']:.2f})")
    return ranks

def plot_layer_energy(svd_task, task_name="", base_task_name="", args=None):
    layer_names = []
    layer_energy = []

    for name, (_, S, _) in svd_task.items():
        layer_names.append(name)
        layer_energy.append((S ** 2).sum().item())  # Frobenius norm^2

    # Sort by energy (optional)
    order = np.argsort(layer_energy)[::-1]
    layer_names = [layer_names[i] for i in order]
    layer_energy = [layer_energy[i] for i in order]

    plt.figure(figsize=(12, 10))
    plt.bar(range(len(layer_names)), layer_energy)
    plt.xticks(range(len(layer_names)), layer_names, rotation=90)
    plt.ylabel("Sum of squared singular values (ΔW Frobenius norm²)")
    plt.title("Total fine-tuning change per layer")
    plt.tight_layout()
    # plt.show()
    save_dir = f"/cis/home/schaud35/clinical-highmmt/src/analysis/plots/{args.fusion_model}/{base_task_name}/{task_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f"{save_dir}/layer_energy.png")

def plot_singular_values_per_layer(svd_task, task_name="", base_task_name="", args=None):
    for name, (_, S, _) in svd_task.items():
        plt.figure(figsize=(12, 10))
        plt.plot(S.numpy(), marker='o')
        plt.yscale('log')  # often log scale helps
        plt.xlabel("Singular value index")
        plt.ylabel("Magnitude (log scale)")
        plt.title(f"Singular values - {name}")
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        save_dir = f"/cis/home/schaud35/clinical-highmmt/src/analysis/plots/{args.fusion_model}/{base_task_name}/{task_name}/{name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}/singular_values.png")

def compare_singular_values(svd_results, task_names=[], base_task_name="", suffix="", args=None):
    if not task_names:
        return  # nothing to compare
    # Get & of all keys of tasks in task_set
    task_keys = set.intersection(*[set(svd_results[name].keys()) for name in task_names])
    for name in task_keys:
        plt.figure(figsize=(12, 10))
        for task_name in task_names:
            S = svd_results[task_name][name][1]
            plt.plot(S.numpy(), label=task_name)
        plt.yscale('log')
        plt.title(f"Singular values comparison: {name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        save_dir = f"/cis/home/schaud35/clinical-highmmt/src/analysis/plots/{args.fusion_model}/{base_task_name}/compare_singular_values{suffix}/{name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}/compare_singular_values.png")

def plot_rank_energy_ratio_per_layer(svd_task, task_name="", base_task_name="", args=None):
    layer_names = []
    rank_ratio = []
    for name, (_, S, _) in svd_task.items():
        r90 = (torch.cumsum(S**2, dim=0) / (S**2).sum()).numpy()
        rank_ratio.append(np.argmax(r90 >= 0.9))  # rank to capture 90% energy
        layer_names.append(name)

    plt.figure(figsize=(12, 10))
    plt.bar(range(len(layer_names)), rank_ratio)
    plt.xticks(range(len(layer_names)), layer_names, rotation=90)
    plt.ylabel("Rank for 90% of energy")
    plt.title("Effective rank of fine-tuning update per layer")
    plt.tight_layout()
    # plt.show()
    save_dir = f"/cis/home/schaud35/clinical-highmmt/src/analysis/plots/{args.fusion_model}/{base_task_name}/{task_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f"{save_dir}/rank_energy_ratio_per_layer.png")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    base_task = args.base_task
    base_task_mod = args.base_task_mods
    new_task_mod = args.new_task_mods

    ihm = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/mimic_iv_{base_task}_{new_task_mod}.pt')
    ihm_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/mimic_iv_{base_task}_{new_task_mod}_{base_task.upper()}_encoder.pt')
    los_ft = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/{base_task}/{base_task_mod}/mimic_iv_los_{new_task_mod}_ft_from_{base_task}.pt')
    los_ft_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/{base_task}/{base_task_mod}/mimic_iv_los_{new_task_mod}_ft_from_{base_task}_LOS_encoder.pt')
    los_lora = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/{base_task}/{base_task_mod}/mimic_iv_los_{new_task_mod}_lora_from_{base_task}.pt')
    los_lora_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/{base_task}/{base_task_mod}/mimic_iv_los_{new_task_mod}_lora_from_{base_task}_LOS_encoder.pt')
    los = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/mimic_iv_los_{new_task_mod}.pt')
    los_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/mimic_iv_los_{new_task_mod}_LOS_encoder.pt')
    pheno_ft = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/{base_task}/{base_task_mod}/mimic_iv_pheno_{new_task_mod}_ft_from_{base_task}.pt')
    pheno_ft_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/{base_task}/{base_task_mod}/mimic_iv_pheno_{new_task_mod}_ft_from_{base_task}_PHENO_encoder.pt')
    pheno_lora = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/{base_task}/{base_task_mod}/mimic_iv_pheno_{new_task_mod}_lora_from_{base_task}.pt')
    pheno_lora_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/{base_task}/{base_task_mod}/mimic_iv_pheno_{new_task_mod}_lora_from_{base_task}_PHENO_encoder.pt')
    pheno = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/mimic_iv_pheno_{new_task_mod}.pt')
    pheno_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/mimic_iv_pheno_{new_task_mod}_PHENO_encoder.pt')
    mortality = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/mimic_iv_mortality_T1-T2-T3-T4-T5.pt')
    mortality_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/mimic_iv_mortality_T1-T2-T3-T4-T5_MOR_encoder.pt')
    readmission = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/mimic_iv_readmission_T1-T2-T3-T4-T5.pt')
    readmission_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/mimic_iv_readmission_T1-T2-T3-T4-T5_RAD_encoder.pt')
    ihm_los_pheno = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/mimic_iv_ihm-los-pheno_{new_task_mod}.pt')
    ihm_los_pheno_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{args.fusion_model}/mimic_iv_ihm-los-pheno_{new_task_mod}_IHM_encoder.pt')
    # for it in [('ihm', ihm, ihm_enc, False, False), ('los', los_ft, los_ft_enc, True, False), ('los', los_lora, los_lora_enc, False, True), ('los', los, los_enc, False, False), ('pheno', pheno_ft, pheno_ft_enc, True, False), ('pheno', pheno_lora, pheno_lora_enc, False, True), ('pheno', pheno, pheno_enc, False, False)]:
    # for it in [('pheno', los, los_enc, False, False)]:
        # print(f"Evaluating {it[0]} model...")
        # args.fine_tune = it[-2]
        # args.lora = it[-1]
        # args.task = it[0]
        # evaluate_model(args, model=it[1].cuda(), encoder=it[2].cuda(), device='cuda')
    # import pdb; pdb.set_trace()
    
    # Create composite: IHM early layers + LOS late layers
    # composite_forward, cleanup = create_composite_model(
    #     ihm, los,
    #     cutoff_layer_source=['modality_layers.TS_IHM.0.2', 'modality_layers.Text_IHM.0.2', 'modality_layers.CXR_IHM.0.2'],  # Extract from IHM here
    #     start_layer_target=['cross_layers.0.0']  # Inject into LOS here
    # )
    # evaluate_model(args, model=ihm.cuda(), encoder=ihm_enc.cuda(), device='cuda', custom_forward=composite_forward)
    rank_cutoff = 0.001

    svd_los = layerwise_svd(los)
    los_ranks = print_layerwise_ranks(los, tol=rank_cutoff)
    df_los = pd.DataFrame({    "layer": list(los_ranks.keys(
    )),    "rank": [v["rank"] for v in los_ranks.values()]})
    df_los.to_csv(f'los_ranks.csv')

    # svd_ihm_enc = layerwise_svd(ihm_enc)
    # ihm_enc_ranks = print_layerwise_ranks(ihm_enc, tol=rank_cutoff)
    # df_ihm_enc = pd.DataFrame({    "layer": list(ihm_enc_ranks.keys(
    # )),    "rank": [v["rank"] for v in ihm_enc_ranks.values()]})
    # df_ihm_enc.to_csv(f'ihm_enc_ranks.csv')
    
    svd_los_enc = layerwise_svd(los_enc)
    los_enc_ranks = print_layerwise_ranks(los_enc, tol=rank_cutoff)
    df_los_enc = pd.DataFrame({    "layer": list(los_enc_ranks.keys(
    )),    "rank": [v["rank"] for v in los_enc_ranks.values()]})
    df_los_enc.to_csv(f'los_enc_ranks.csv')

    # svd_pheno_enc = layerwise_svd(pheno_enc)
    # pheno_enc_ranks = print_layerwise_ranks(pheno_enc, tol=rank_cutoff)
    # df_pheno_enc = pd.DataFrame({    "layer": list(pheno_enc_ranks.keys(
    # )),    "rank": [v["rank"] for v in pheno_enc_ranks.values()]})
    # df_pheno_enc.to_csv(f'pheno_enc_ranks.csv')
    import pdb; pdb.set_trace()
    
    svd_ihm_los_pheno = layerwise_svd(ihm_los_pheno)
    ihm_los_pheno_ranks = print_layerwise_ranks(ihm_los_pheno, tol=rank_cutoff)
    df_ihm_los_pheno = pd.DataFrame({    "layer": list(ihm_los_pheno_ranks.keys(
    )),    "rank": [v["rank"] for v in ihm_los_pheno_ranks.values()]})
    df_ihm_los_pheno.to_csv(f'ihm_los_pheno_ranks.csv')
    
    svd_ihm_los_pheno_enc = layerwise_svd(ihm_los_pheno_enc)
    ihm_los_pheno_enc_ranks = print_layerwise_ranks(ihm_los_pheno_enc, tol=rank_cutoff)
    df_ihm_los_pheno_enc = pd.DataFrame({    "layer": list(ihm_los_pheno_enc_ranks.keys(
    )),    "rank": [v["rank"] for v in ihm_los_pheno_enc_ranks.values()]})
    df_ihm_los_pheno_enc.to_csv(f'ihm_los_pheno_enc_ranks.csv')
    import pdb; pdb.set_trace()

    svd_svd_mortality = layerwise_svd(mortality)
    mortality_ranks = print_layerwise_ranks(mortality, tol=rank_cutoff)
    df_mortality = pd.DataFrame({    "layer": list(mortality_ranks.keys(
    )),    "rank": [v["rank"] for v in mortality_ranks.values()]})
    df_mortality.to_csv(f'mortality_ranks.csv')

    svd_svd_readmission = layerwise_svd(readmission)
    readmission_ranks = print_layerwise_ranks(readmission, tol=rank_cutoff)
    df_readmission = pd.DataFrame({    "layer": list(readmission_ranks.keys(
    )),    "rank": [v["rank"] for v in readmission_ranks.values()]})
    df_readmission.to_csv(f'readmission_ranks.csv')

    svd_los = layerwise_svd(los)
    los_ranks = print_layerwise_ranks(los, tol=rank_cutoff)
    # los_enc_ranks = print_layerwise_ranks(los_enc, tol=rank_cutoff)
    los_k = {}
    for n, (u,s,v) in svd_los.items():
        if n in los_ranks:
            los_k[n] = torch.matmul(u[:, :los_ranks[n]['rank']], torch.matmul(torch.diag(s[:los_ranks[n]['rank']]), v[:los_ranks[n]['rank'], :]))
        else:
            los_k[n] = torch.matmul(u, torch.matmul(torch.diag(s), v))
    copy_weights(los_k, los)
    args.fine_tune=False
    args.lora=False
    args.task='los'
    evaluate_model(args, model=los.cuda(), encoder=los_enc.cuda(), device='cuda', inter_dir=f'lower_rank_{rank_cutoff}_los')
    
    svd_ihm = layerwise_svd(ihm)
    ihm_ranks = print_layerwise_ranks(ihm, tol=rank_cutoff)
    df_ihm = pd.DataFrame({    "layer": list(ihm_ranks.keys(
    )),    "rank": [v["rank"] for v in ihm_ranks.values()]})
    ihm_k = {}
    for n, (u,s,v) in svd_ihm.items():
        if n in ihm_ranks:
            ihm_k[n] = torch.matmul(u[:, :ihm_ranks[n]['rank']], torch.matmul(torch.diag(s[:ihm_ranks[n]['rank']]), v[:ihm_ranks[n]['rank'], :]))
        else:
            ihm_k[n] = torch.matmul(u, torch.matmul(torch.diag(s), v))
    copy_weights(ihm_k, ihm)
    args.fine_tune=False
    args.lora=False
    args.task='ihm'
    evaluate_model(args, model=ihm.cuda(), encoder=ihm_enc.cuda(), device='cuda', inter_dir=f'lower_rank_{rank_cutoff}_ihm')
    
    
    # svd_los_ft = layerwise_svd(los_ft)
    # los_ft_ranks = print_layerwise_ranks(los_ft, tol=rank_cutoff)
    # los_ft_k = {}
    # for n, (u,s,v) in svd_los_ft.items():
    #     if n in los_ft_ranks:
    #         los_ft_k[n] = torch.matmul(u[:, :los_ft_ranks[n]['rank']], torch.matmul(torch.diag(s[:los_ft_ranks[n]['rank']]), v[:los_ft_ranks[n]['rank'], :]))
    #     else:
    #         los_ft_k[n] = torch.matmul(u, torch.matmul(torch.diag(s), v))
    # copy_weights(los_ft_k, los_ft)
    # args.fine_tune=True
    # args.lora=False
    # args.task='los'
    # evaluate_model(args, model=los_ft.cuda(), encoder=los_ft_enc.cuda(), device='cuda', inter_dir=f'lower_rank_{rank_cutoff}_los_ft')
    
    los_to_ihm, ihm_to_los, U_ihm_los, S_ihm_los, Vh_ihm_los = layerwise_concat_svd(ihm, los, tol=rank_cutoff)
    copy_weights(ihm_to_los, los)
    copy_weights(los_to_ihm, ihm)
    args.fine_tune=False
    args.lora=False
    args.task='los'
    evaluate_model(args, model=los.cuda(), encoder=los_enc.cuda(), device='cuda', inter_dir=f'lower_rank_{rank_cutoff}_eigen_ihm_to_los')
    args.task='ihm'
    evaluate_model(args, model=ihm.cuda(), encoder=ihm_enc.cuda(), device='cuda', inter_dir=f'lower_rank_{rank_cutoff}_eigen_los_to_ihm')
    import pdb; pdb.set_trace()


    start_copy_layer = 10
    end_copy_layer = 20
    copy_weights(ihm, los, start_copy_layer, end_copy_layer)
    args.fine_tune=False
    args.lora=False
    args.task='los'
    evaluate_model(args, model=los.cuda(), encoder=los_enc.cuda(), device='cuda', inter_dir=f'composite_{start_copy_layer}_{end_copy_layer}_ihm_in_los')
    
    ihm_ranks = print_layerwise_ranks(ihm)
    concat_ranks = print_layerwise_concat_ranks(ihm, los)
    import pdb; pdb.set_trace()
    task_vectors = {
        # "ihm_TS-Text-CXR": get_task_vectors(ihm, ihm),
        f"los_ft_{new_task_mod}/{base_task}_{base_task_mod}": get_task_vectors(ihm, los_ft),
        f"los_lora_{new_task_mod}/{base_task}_{base_task_mod}": los_lora,
        f"los_{new_task_mod}/{base_task}_{base_task_mod}": get_task_vectors(ihm, los),
        f"pheno_ft_{new_task_mod}/{base_task}_{base_task_mod}": get_task_vectors(ihm, pheno_ft),
        f"pheno_lora_{new_task_mod}/{base_task}_{base_task_mod}": pheno_lora,
        f"pheno_{new_task_mod}/{base_task}_{base_task_mod}": get_task_vectors(ihm, pheno),
    }

    svd_results = {}
    for task_name, tv in task_vectors.items():
        source_task, target_task = task_name.split('/')
        if 'lora' in task_name:
            svd_results[source_task] = layerwise_svd(tv, rank=None, lora_only=True)
        else:
            svd_results[source_task] = layerwise_svd(tv, rank=None)
        plot_layer_energy(svd_results[source_task], task_name=source_task, base_task_name=target_task, args=args)
        plot_singular_values_per_layer(svd_results[source_task], task_name=source_task, base_task_name=target_task, args=args)
        plot_rank_energy_ratio_per_layer(svd_results[source_task], task_name=source_task, base_task_name=target_task, args=args)

    compare_singular_values(svd_results, task_names=[f"los_ft_{new_task_mod}", f"los_{new_task_mod}", f"pheno_ft_{new_task_mod}", f"pheno_{new_task_mod}"], base_task_name=f"{base_task}_{base_task_mod}", suffix=f'_{new_task_mod}', args=args)
    compare_singular_values(svd_results, task_names=[f"los_lora_{new_task_mod}", f"pheno_lora_{new_task_mod}"], base_task_name=f"{base_task}_{base_task_mod}", suffix=f"_{new_task_mod}_lora", args=args)
    
    import pdb; pdb.set_trace()