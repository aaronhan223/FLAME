import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
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

def layerwise_svd(task_vector, rank=None, lora_only=False):
    svd_results = {}
    for name, delta in (task_vector.items() if not lora_only else task_vector.state_dict().items()):
        if lora_only and "lora_" not in name:
            continue
        if delta.ndim < 2:  # skip biases, layernorm weights etc.
            continue
        
        # Flatten into matrix for SVD
        delta_2d = delta.detach().cpu()
        if delta_2d.ndim > 2:
            print(name, delta_2d.shape)
            delta_2d = delta_2d.view(delta_2d.size(0), -1)
        
        # Compute SVD
        U, S, Vh = torch.linalg.svd(delta_2d, full_matrices=False)
        if rank is not None:
            U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]
        
        svd_results[name] = (U, S, Vh)
        print(f"{name}: shape={delta.shape}, top singular values={S[:5]}")
    
    return svd_results

def plot_layer_energy(svd_task, task_name="", base_task_name=""):
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
    save_dir = f"/cis/home/schaud35/clinical-highmmt/src/analysis/plots/{base_task_name}/{task_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f"{save_dir}/layer_energy.png")

def plot_singular_values_per_layer(svd_task, task_name="", base_task_name=""):
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
        save_dir = f"/cis/home/schaud35/clinical-highmmt/src/analysis/plots/{base_task_name}/{task_name}/{name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}/singular_values.png")

def compare_singular_values(svd_results, task_names=[], base_task_name="", suffix=""):
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
        save_dir = f"/cis/home/schaud35/clinical-highmmt/src/analysis/plots/{base_task_name}/compare_singular_values{suffix}/{name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}/compare_singular_values.png")

def plot_rank_energy_ratio_per_layer(svd_task, task_name="", base_task_name=""):
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
    save_dir = f"/cis/home/schaud35/clinical-highmmt/src/analysis/plots/{base_task_name}/{task_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f"{save_dir}/rank_energy_ratio_per_layer.png")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    base_task = args.base_task
    base_task_mod = args.base_task_mods
    new_task_mod = args.new_task_mods

    ihm = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/mimic_iv_{base_task}_{new_task_mod}.pt')
    ihm_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/mimic_iv_{base_task}_{new_task_mod}_{base_task.upper()}_encoder.pt')
    los_ft = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{base_task}/{base_task_mod}/mimic_iv_los_{new_task_mod}_ft_from_{base_task}.pt')
    los_ft_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{base_task}/{base_task_mod}/mimic_iv_los_{new_task_mod}_ft_from_{base_task}_LOS_encoder.pt')
    los_lora = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{base_task}/{base_task_mod}/mimic_iv_los_{new_task_mod}_lora_from_{base_task}.pt')
    los_lora_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{base_task}/{base_task_mod}/mimic_iv_los_{new_task_mod}_lora_from_{base_task}_LOS_encoder.pt')
    los = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/mimic_iv_los_{new_task_mod}.pt')
    los_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/mimic_iv_los_{new_task_mod}_LOS_encoder.pt')
    pheno_ft = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{base_task}/{base_task_mod}/mimic_iv_pheno_{new_task_mod}_ft_from_{base_task}.pt')
    pheno_ft_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{base_task}/{base_task_mod}/mimic_iv_pheno_{new_task_mod}_ft_from_{base_task}_PHENO_encoder.pt')
    pheno_lora = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{base_task}/{base_task_mod}/mimic_iv_pheno_{new_task_mod}_lora_from_{base_task}.pt')
    pheno_lora_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/{base_task}/{base_task_mod}/mimic_iv_pheno_{new_task_mod}_lora_from_{base_task}_PHENO_encoder.pt')
    pheno = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/mimic_iv_pheno_{new_task_mod}.pt')
    pheno_enc = torch.load(f'/cis/home/schaud35/clinical-highmmt/src/checkpoints/mimic_iv_pheno_{new_task_mod}_PHENO_encoder.pt')
    for it in [('ihm', ihm, ihm_enc, False, False), ('los', los_ft, los_ft_enc, True, False), ('los', los_lora, los_lora_enc, False, True), ('los', los, los_enc, False, False), ('pheno', pheno_ft, pheno_ft_enc, True, False), ('pheno', pheno_lora, pheno_lora_enc, False, True), ('pheno', pheno, pheno_enc, False, False)]:
        print(f"Evaluating {it[0]} model...")
        args.fine_tune = it[-2]
        args.lora = it[-1]
        args.task = it[0]
        evaluate_model(args, model=it[1].cuda(), encoder=it[2].cuda(), device='cuda')
    # import pdb; pdb.set_trace()
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
        plot_layer_energy(svd_results[source_task], task_name=source_task, base_task_name=target_task)
        plot_singular_values_per_layer(svd_results[source_task], task_name=source_task, base_task_name=target_task)
        plot_rank_energy_ratio_per_layer(svd_results[source_task], task_name=source_task, base_task_name=target_task)

    compare_singular_values(svd_results, task_names=[f"los_ft_{new_task_mod}", f"los_{new_task_mod}", f"pheno_ft_{new_task_mod}", f"pheno_{new_task_mod}"], base_task_name=f"{base_task}_{base_task_mod}", suffix=f'_{new_task_mod}')
    compare_singular_values(svd_results, task_names=[f"los_lora_{new_task_mod}", f"pheno_lora_{new_task_mod}"], base_task_name=f"{base_task}_{base_task_mod}", suffix=f"_{new_task_mod}_lora")
    
    import pdb; pdb.set_trace()