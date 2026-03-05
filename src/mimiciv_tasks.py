import sys
import os
import argparse
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from src.crossattnperceiver import MultiModalityPerceiver, InputModality, PerceiverWrapper, CrossAttnTransformer
from src.fusemoe import *
from src.mimiciv_task_setup import setup_tasks_and_modalities
from src.train_structure_multitask_mimic import train
from src.encoders import ModalityEncoders, FSEncoder, EMBEDEncoder
from src.shared_encoders import TimeQueryEncoder
# from src.shared_encoders import ModalityEncoders, FSEncoder, TimeQueryEncoder
from src.utils import create_directory, dump_pickle
from src.preprocess.preprocess_eicu import *
import torch
from accelerate import Accelerator
torch.multiprocessing.set_sharing_strategy('file_system')
from src.datasets.mimic.get_data_mimic_iv import data_prepare as prepare_mimic
from src.get_data_eicu import data_prepare as prepare_eicu
from src.datasets.embed.get_data_embed import data_prepare as prepare_embed
from peft import get_peft_model, LoraConfig, TaskType
from transformers import set_seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    parser.add_argument(
        "--embed_path", type=str, default='/export/io79/data/schaud35/datasets/EMBED', help="Path to pre-extracted embeddings for each modality and task in EMBED, required if --use_pt_text_embeddings is set."
    )
    parser.add_argument("--ihm_mod", type=str, default='', help="Modality compoenents for IHM task.")
    parser.add_argument("--los_mod", type=str, default='', help="Modality compoenents for LOS task.")
    parser.add_argument("--pheno_mod", type=str, default='', help="Modality compoenents for PHENO task.")
    parser.add_argument("--rad_mod", type=str, default='', help="Modality compoenents for readmission task.")
    parser.add_argument("--mor_mod", type=str, default='', help="Modality compoenents for mortality task.")
    parser.add_argument("--birads_mod", type=str, default='', help="Modality compoenents for birads task.")
    parser.add_argument("--risk_mod", type=str, default='', help="Modality compoenents for cancer risk prediction task.")
    parser.add_argument("--density_mod", type=str, default='', help="Modality compoenents for tissue density prediction task.")

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
        "--train_bs_embed",
        type=int,
        default=8,
        help="Batch size for the embed training dataloader.",
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
    parser.add_argument("--tt_max_eicu", default=1, type=int, help="max time for eicu data.")
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
    parser.add_argument('--mixup_level', type=str, default='batch', help='mixup level: batch or batch_seq or batch_seq_feature')
    parser.add_argument('--cross_method', type=str, default='moe', help='cross attention method: moe or self_cross or hme')

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
    parser.add_argument('--results_dir', type=str, default='/cis/home/schaud35/clinical-highmmt/src/results', help='Directory to store results') 
    parser.add_argument('--fusion_model', type=str, default='multimodalityperceiver', help='Fusion model to use, Perceiver or CrossAttnTransformer')
    parser.add_argument('--linear_probe', action='store_true')
    parser.add_argument('--shared_modality_encoders', action='store_true', help='Use shared modality encoders across tasks')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging for train/val/test metrics.')
    parser.add_argument('--wandb_project', type=str, default='clinical-highmmt', help='Weights & Biases project name.')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Optional Weights & Biases run name.')
    parser.add_argument("--num_of_experts", nargs='*', type=int, help="number of MLPs in MoE, for HME need to specify each level")
    parser.add_argument("--top_k", nargs='*', type=int, help="the number of experts finally combined together for joint and permod routers")
    parser.add_argument("--router_type", default='joint', type=str, help="all router types: joint, permod, disjoint")
    parser.add_argument("--gating_function", nargs='*', type=str, help="all gating functions: softmax, laplace, gaussian, enter at least one")
    parser.add_argument("--modality_drop_rate", default=0.0, type=float, help="Probability of dropping each modality from indict before model forward pass (keeps at least one). 0.0 = no dropping.")
    parser.add_argument("--multitask_moe", action='store_true', help="Whether to use the multitask MoE implementation in src/fusemoe_multitask.py instead of the original MoE implementation in src/sparse_moe.py. The multitask MoE allows for different gating and expert configurations per task, while the original MoE uses the same gating and expert configuration for all tasks.")
    args = parser.parse_args()
    return args


def loadBert(args,device):
    if args.model_name!=None:
        if args.model_name== 'BioBert':
            tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            BioBert=AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif args.model_name=="bioRoberta":
            config = AutoConfig.from_pretrained("allenai/biomed_roberta_base", num_labels=args.num_labels)
            tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
            BioBert = AutoModel.from_pretrained("allenai/biomed_roberta_base")
        elif args.model_name== "Bert":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            BioBert = BertModel.from_pretrained("bert-base-uncased")
        elif args.model_name== "bioLongformer":
            tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
            BioBert = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
        else:
            raise ValueError("model_name should be BioBert,bioRoberta,bioLongformer or Bert")
    else:
        if args.model_path!=None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            BioBert = AutoModel.from_pretrained(args.model_path)
        else:
            raise ValueError("provide either model_name or model_path")

    BioBert = BioBert.to(device)
    BioBertConfig = BioBert.config
    return BioBert, BioBertConfig, tokenizer

# Function to replace Sequential modules in target_modules with their submodules
def update_target_modules_for_sequential(model, target_modules):
    updated_target_modules = set()

    for module_name in target_modules:
        # Check if the module is part of a Sequential block
        if isinstance(dict(model.named_modules()).get(module_name.split('.')[0]), torch.nn.Sequential):
            sequential_module = dict(model.named_modules()).get(module_name.split('.')[0])

            # Iterate through the submodules inside the Sequential block
            for sub_name, sub_module in sequential_module.named_children():
                # Add the submodule layer to the updated_target_modules if it's a valid LoRA layer
                if isinstance(sub_module, torch.nn.Linear) or isinstance(sub_module, torch.nn.Conv1d):
                    full_name = f"{module_name.split('.')[0]}.{sub_name}.weight"
                    updated_target_modules.add(full_name)
        else:
            # If the module is not sequential, just keep it in the updated_target_modules
            updated_target_modules.add(module_name)

    return updated_target_modules

def main():
    args = parse_args()
    set_seed(args.seed)
    if args.fp16:
        args.mixed_precision = "fp16"
    else:
        args.mixed_precision = "no"
    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=args.cpu)
    device = accelerator.device

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
        'mortality-readmission_mod': args.mor_mod+'_'+args.rad_mod,
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

    modalities = set()
    modeltype = {}
    # for t in args.task.split("-"):
    #     modeltype[t] = '_'.join(sorted(getattr(args, f"{t}_mod").split("-")))
    if len(args.ihm_mod) != 0 and 'ihm' in args.task.split("-"):
        modeltype['ihm'] = '_'.join(sorted(args.ihm_mod.split("-")))
        for e in args.ihm_mod.split("-"):
            modalities.add(e)
    if len(args.los_mod) != 0 and 'los' in args.task.split("-"):
        modeltype['los'] = '_'.join(sorted(args.los_mod.split("-")))
        for e in args.los_mod.split("-"):
            modalities.add(e)
    if len(args.pheno_mod) != 0 and 'pheno' in args.task.split("-"):
        modeltype['pheno'] = '_'.join(sorted(args.pheno_mod.split("-")))
        for e in args.pheno_mod.split("-"):
            modalities.add(e)
    if len(args.rad_mod) != 0 and 'readmission' in args.task.split("-"):
        modeltype['readmission'] = '_'.join(sorted(args.rad_mod.split("-")))
        for e in args.rad_mod.split("-"):
            modalities.add(e)
    if len(args.mor_mod) != 0 and 'mortality' in args.task.split("-"):
        modeltype['mortality'] = '_'.join(sorted(args.mor_mod.split("-")))
        for e in args.mor_mod.split("-"):
            modalities.add(e)
    if len(args.birads_mod) != 0 and 'birads' in args.task.split("-"):
        modeltype['birads'] = '_'.join(sorted(args.birads_mod.split("-")))
        for e in args.birads_mod.split("-"):
            modalities.add(e)
    if len(args.risk_mod) != 0 and 'risk' in args.task.split("-"):
        modeltype['risk'] = '_'.join(sorted(args.risk_mod.split("-")))
        for e in args.risk_mod.split("-"):
            modalities.add(e)
    if len(args.density_mod) != 0 and 'density' in args.task.split("-"):
        modeltype['density'] = '_'.join(sorted(args.density_mod.split("-")))
        for e in args.density_mod.split("-"):
            modalities.add(e)
    
        
    # modeltype = ''
    # modals = [*modalities]
    # modals.sort()
    # for m in modals:
    #     modeltype = modeltype + m + '_'
    # modeltype = modeltype[:-1]

    # if len(args.rad_mod) != 0 and 'readmission' in args.task:
    #     for e in args.rad_mod.split("-"):
    #         modalities.add(e)
    # if len(args.mor_mod) != 0 and 'mortality' in args.task:
    #     for e in args.mor_mod.split("-"):
    #         modalities.add(e)
    # modalities = [*modalities]

    if 'Text' in modalities:
        BioBert, BioBertConfig, tokenizer = loadBert(args, device)
    else:
        tokenizer = None
        BioBert = None

    (
        all_train,
        all_valid,
        all_test,
        criterion,
        modalities_per_task,
        train_weights,
        all_encoders,
        logits,
        all_modalities,
    ) = setup_tasks_and_modalities(
        args=args,
        device=device,
        tokenizer=tokenizer,
        modeltype=modeltype,
        modalities=modalities,
        BioBert=BioBert,
    )
    
    # TODO: each feature a modality? clustering feature?
    # TODO: should we keep feature specific encoders?
    # import pdb; pdb.set_trace()
    
    perceiver_mod = []
    if not args.shared_modality_encoders:
        for t in modalities_per_task:
            for m in t:
                perceiver_mod.append(all_modalities[m])
    else:
        # Common modalities across all tasks
        perceiver_mod = []
        shared_modalities = set([m for tm in task_mods_dict[task_mod_key].split('_') for m in tm.split('-')])
        for m in shared_modalities:
            perceiver_mod.append(all_modalities[m])
    # # modalities_per_task = [[i.split('_')[0] for i in j] for j in modalities_per_task]
    
    if args.fusion_model=="multimodalityperceiver":
        model = MultiModalityPerceiver(
            modalities=perceiver_mod,
            depth=1,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
            num_latents=20,
            # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=args.perceiver_dim,  # latent dimension
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=6,  # number of heads for latent self attention, 8
            cross_dim_head=64,
            latent_dim_head=64,
            num_classes=1,  # output number of classes
            attn_dropout=0.,
            ff_dropout=0.,
            #embed=True,
            weight_tie_layers=True,
            num_latent_blocks_per_layer=1,
            cross_depth=1# Note that this parameter is 1 in the original Lucidrain implementation
            # whether to weight tie layers (optional, as indicated in the diagram)
        ).to(device)
    elif args.fusion_model in ["crossattntransformer", "crossattntransformer_wo_residual"]:
        model = CrossAttnTransformer(
            modalities=perceiver_mod,
            depth=1,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
            num_latents=20,
            # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=args.perceiver_dim,  # latent dimension
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=6,  # number of heads for latent self attention, 8
            cross_dim_head=64,
            latent_dim_head=64,
            num_classes=1,  # output number of classes
            attn_dropout=0.,
            ff_dropout=0.,
            #embed=True,
            weight_tie_layers=True,
            num_latent_blocks_per_layer=1,
            cross_depth=1# Note that this parameter is 1 in the original Lucidrain implementation
            # whether to weight tie layers (optional, as indicated in the diagram)
        ).to(device)
    elif args.fusion_model in ['fusemoe']:
        model = MULTCrossModel(
            args,
            device,
            modeltype=task_mods_dict[task_mod_key],
            modalities=perceiver_mod,
            modalities_per_task=modalities_per_task,
            num_classes=1
        ).to(device)
    else:
        raise ValueError("fusion_model should be multimodalityperceiver or crossattntransformer")
    
    model.to_logitslist = logits.to(device)
    # import pdb; pdb.set_trace()
    
    if args.fine_tune or args.lora:
        # Load the saved model checkpoint
        checkpoint = torch.load(f'./checkpoints/{args.fusion_model}/mimic_iv_{args.base_task}_{args.base_task_mods}.pt', map_location=device)
        for ii in range(len(modalities_per_task)):
            task = modalities_per_task[int(ii)][0].split('_')[1]
            enc_checkpoint = torch.load(f'./checkpoints/{args.fusion_model}/mimic_iv_{args.base_task}_{args.base_task_mods}_{task}_encoder.pt', map_location=device)
        
        # This will be the state_dict (either entire checkpoint or nested in a dict)
        pretrained_state_dict = checkpoint.state_dict()
        pretrained_enc_state_dict = enc_checkpoint.state_dict()

        # Get the current model's state_dict
        model_state_dict = model.state_dict()

        # Filter the pretrained weights to only those that match in shape and name
        compatible_weights = {}
        for k, v in pretrained_state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                compatible_weights[k] = v

        # Report mismatches if desired
        mismatched_keys = [k for k in pretrained_state_dict if k not in compatible_weights]
        if mismatched_keys:
            print("Skipping incompatible or missing keys:")
            for k in mismatched_keys:
                ckpt_shape = pretrained_state_dict[k].shape
                model_value = model_state_dict.get(k, None)
                model_shape = model_value.shape if model_value is not None else 'missing'
                print(f" - {k} (checkpoint shape: {ckpt_shape}, model shape: {model_shape})")

        # Load compatible weights
        model_state_dict.update(compatible_weights)
        model.load_state_dict(model_state_dict)

        print("Successfully loaded compatible model weights.")
        
        for ii in range(len(modalities_per_task)):
            task = modalities_per_task[int(ii)][0].split('_')[1]
            enc_state_dict = all_encoders[task.upper()].state_dict()
            compatible_enc_weights = {}
            
            for k, v in pretrained_enc_state_dict.items():
                if k in enc_state_dict and enc_state_dict[k].shape == v.shape:
                    compatible_enc_weights[k] = v
            
            mismatched_enc_keys = [k for k in pretrained_enc_state_dict if k not in compatible_enc_weights]
            if mismatched_enc_keys:
                print(f"Skipping incompatible or missing encoder keys for {t}:")
                for k in mismatched_enc_keys:
                    ckpt_shape = pretrained_enc_state_dict[k].shape
                    model_value = enc_state_dict.get(k, None)
                    model_shape = model_value.shape if model_value is not None else 'missing'
                    print(f" - {k} (checkpoint shape: {ckpt_shape}, model shape: {model_shape})")
            
            enc_state_dict.update(compatible_enc_weights)
            all_encoders[task.upper()].load_state_dict(enc_state_dict)
            # Freeze only the parameters that were matched and loaded
            for name, param in all_encoders[task.upper()].named_parameters():
                if name in compatible_enc_weights: 
                    param.requires_grad = False
        print("Successfully loaded compatible encoder weights.")
    
    if args.linear_probe:
        model = torch.load(f'./checkpoints/{args.fusion_model}/mimic_iv_{args.base_task}_{args.base_task_mods}.pt', map_location=device)
        model.to_logitslist = logits.to(device)
        for ii in range(len(modalities_per_task)):
            task = modalities_per_task[int(ii)][0].split('_')[1]
            if args.base_task==task.lower():
                all_encoders[task] = torch.load(f'./checkpoints/{args.fusion_model}/mimic_iv_{args.base_task}_{args.base_task_mods}_{task}_encoder.pt', map_location=device)
        for name, param in model.named_parameters():
            if 'to_logits' not in name:
                param.requires_grad = False
        for task in all_encoders:
            for name, param in all_encoders[task].named_parameters():
                param.requires_grad = False

    if args.lora:
        # Wrap model to make it PEFT-compatible
        wrapped_model = PerceiverWrapper(model)

        # 1. Get loaded layers' names
        loaded_layer_names = set(k.rsplit('.', 1)[0] for k in compatible_weights)

        # 2. Define which types of submodules you want to LoRA (e.g., projection layers)
        possible_lora_targets = [
            "to_q",   # For attention query weights
            "to_kv",  # For attention key/value weights
            "to_out", # For output projection
            "net"     # For intermediate feedforward weights
        ]

        # 3. Find which of those submodules appear in the loaded layers
        target_modules = set()
        for name in loaded_layer_names:
            for sub in possible_lora_targets:
                if sub in name:
                    target_modules.add(name)
        print("Target LoRA modules:", target_modules)
        target_modules = update_target_modules_for_sequential(model, target_modules)
        print("Target LoRA modules:", target_modules)
        
        # 4. Define LoRA config only for those submodules
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # change based on your task
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=list(target_modules),
        )

        # 5. Apply LoRA

        model = get_peft_model(wrapped_model, lora_config)
    
    # print([(n,p.shape) for n,p in model.named_parameters()])
    # exit()
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
    if args.lora:
        savedir = f'./checkpoints/{args.fusion_model}/{args.base_task}/{args.base_task_mods}/{args.task}_{task_mods_dict[task_mod_key]}_lora_from_{args.base_task}.pt'
        os.makedirs(os.path.dirname(savedir), exist_ok=True)
    elif args.fine_tune:
        savedir = f'./checkpoints/{args.fusion_model}/{args.base_task}/{args.base_task_mods}/{args.task}_{task_mods_dict[task_mod_key]}_ft_from_{args.base_task}.pt'
        os.makedirs(os.path.dirname(savedir), exist_ok=True)
    elif args.linear_probe:
        savedir = f'./checkpoints/{args.fusion_model}/{args.base_task}/{args.base_task_mods}/{args.task}_{task_mods_dict[task_mod_key]}_linear_probe_from_{args.base_task}.pt'
        os.makedirs(os.path.dirname(savedir), exist_ok=True)
    else:
        if args.shared_modality_encoders:
            if args.multitask_moe:
                savedir = f'./checkpoints/flame/multitask/{args.task}/{args.task}_{task_mods_dict[task_mod_key]}_mod_drop_rate_{args.modality_drop_rate}.pt'
            else:
                savedir = f'./checkpoints/{args.fusion_model}/multitask/{args.task}/{args.task}_{task_mods_dict[task_mod_key]}_mod_drop_rate_{args.modality_drop_rate}.pt'
        else:
            savedir = f'./checkpoints/{args.fusion_model}/{args.base_task}/{args.base_task_mods}/{args.task}_{task_mods_dict[task_mod_key]}_mod_drop_rate_{args.modality_drop_rate}.pt'
        os.makedirs(os.path.dirname(savedir), exist_ok=True)
    if args.num_train_epochs>0:
        torch.save(model,savedir)
        for ii in range(len(modalities_per_task)):
            task = modalities_per_task[int(ii)][0].split('_')[1]
            torch.save(all_encoders[task], f'{savedir.split(".pt")[0]}_{task}_encoder.pt')
    
    _ = train(
        model,
        all_train,
        all_valid,
        all_test,
        modalities_per_task,
        savedir,
        args,
        all_encoders,
        setting,
        criterion=criterion,
        lr=0.0008,
        device=device,
        train_weights=train_weights,
        weight_decay=0.001    
    )
    print('Experiment done!')


if __name__ == "__main__":
    import pdb
    main()