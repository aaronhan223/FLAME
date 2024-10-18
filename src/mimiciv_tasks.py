import sys
import os
import argparse
sys.path.insert(1,os.getcwd())
from crossattnperceiver import MultiModalityPerceiver, InputModality
from train_structure_multitask_mimic import train
from encoders import ModalityEncoders
import torch
import pdb

import torch
from accelerate import Accelerator
torch.multiprocessing.set_sharing_strategy('file_system')
from datasets.mimic.get_data_mimic_iv import data_prepare
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

def parse_args():
    parser = argparse.ArgumentParser(description="Alignment text and ts data")
    parser.add_argument(
            "--task", type=str, default="ihm-los-pheno"
        )
    parser.add_argument(
        "--file_path", type=str, default="/cis/home/xhan56/code/Multimodal-Transformer/src/Data/ihm", help="A path to dataset folder"
    )
    parser.add_argument("--ihm_mod", type=str, default='', help="Modality compoenents for IHM task.")
    parser.add_argument("--los_mod", type=str, default='', help="Modality compoenents for LOS task.")
    parser.add_argument("--pheno_mod", type=str, default='', help="Modality compoenents for PHENO task.")

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
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size  for the training dataloader.",
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

args = parse_args()
if args.fp16:
    args.mixed_precision = "fp16"
else:
    args.mixed_precision = "no"
accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=args.cpu)
device = accelerator.device

modalities = set()
if len(args.ihm_mod) != 0:
    for e in args.ihm_mod.split("-"):
        modalities.add(e)
if len(args.los_mod) != 0:
    for e in args.los_mod.split("-"):
        modalities.add(e)
if len(args.pheno_mod) != 0:
    for e in args.pheno_mod.split("-"):
        modalities.add(e)
modalities = [*modalities]
modalities.sort()

modeltype = ''
for m in modalities:
    modeltype = modeltype + m + '_'
modeltype = modeltype[:-1]

if 'Text' in modeltype:
    BioBert, BioBertConfig, tokenizer = loadBert(args, device)
else:
    tokenizer = None

tasks = args.task.split("-")
all_train = []
all_valid = []
all_test = []
criterion = []
modalities_per_task = []
logits = torch.nn.ModuleList()
if 'ihm' in tasks:
    train_ihm, valid_ihm, test_ihm = data_prepare(args=args, task='ihm', tokenizer=tokenizer, modeltype=modeltype)
    all_train.append(train_ihm)
    all_valid.append(valid_ihm)
    all_test.append(test_ihm)
    criterion.append(torch.nn.CrossEntropyLoss())
    ihm_mods = list(map(lambda s: s + '_IHM', args.ihm_mod.split("-")))
    assert len(ihm_mods) > 1, "At least two modalities per task!"
    modalities_per_task.append(ihm_mods)
    logit_dim = len(ihm_mods) * (len(ihm_mods) - 1) * args.perceiver_dim
    logits.append(torch.nn.Sequential(torch.nn.LayerNorm(logit_dim), torch.nn.Linear(logit_dim, 2)))
if 'los' in tasks:
    train_los, valid_los, test_los = data_prepare(args=args, task='los', tokenizer=tokenizer, modeltype=modeltype)
    all_train.append(train_los)
    all_valid.append(valid_los)
    all_test.append(test_los)
    criterion.append(torch.nn.CrossEntropyLoss())
    los_mods = list(map(lambda s: s + '_LOS', args.los_mod.split("-")))
    assert len(los_mods) > 1, "At least two modalities per task!"
    modalities_per_task.append(los_mods)
    logit_dim = len(los_mods) * (len(los_mods) - 1) * args.perceiver_dim
    logits.append(torch.nn.Sequential(torch.nn.LayerNorm(logit_dim), torch.nn.Linear(logit_dim, 2)))
if 'pheno' in tasks:
    train_pheno, valid_pheno, test_pheno = data_prepare(args=args, task='pheno', tokenizer=tokenizer, modeltype=modeltype)
    all_train.append(train_pheno)
    all_valid.append(valid_pheno)
    all_test.append(test_pheno)
    criterion.append(torch.nn.BCEWithLogitsLoss())
    pheno_mods = list(map(lambda s: s + '_PHENO', args.pheno_mod.split("-")))
    assert len(pheno_mods) > 1, "At least two modalities per task!"
    modalities_per_task.append(pheno_mods)
    logit_dim = len(pheno_mods) * (len(pheno_mods) - 1) * args.perceiver_dim
    logits.append(torch.nn.Sequential(torch.nn.LayerNorm(logit_dim), torch.nn.Linear(logit_dim, 25)))

all_modalities = {}
all_modalities['Text_IHM'] = InputModality(
    name='Text_IHM',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['TS_IHM'] = InputModality(
    name='TS_IHM',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['CXR_IHM'] = InputModality(
    name='CXR_IHM',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1.
)
all_modalities['ECG_IHM'] = InputModality(
    name='ECG_IHM',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1.
)
all_modalities['Text_LOS'] = InputModality(
    name='Text_LOS',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['TS_LOS'] = InputModality(
    name='TS_LOS',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['CXR_LOS'] = InputModality(
    name='CXR_LOS',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1.
)
all_modalities['ECG_LOS'] = InputModality(
    name='ECG_LOS',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1.
)
all_modalities['Text_PHENO'] = InputModality(
    name='Text_PHENO',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['TS_PHENO'] = InputModality(
    name='TS_PHENO',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['CXR_PHENO'] = InputModality(
    name='CXR_PHENO',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1.
)
all_modalities['ECG_PHENO'] = InputModality(
    name='ECG_PHENO',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1.
)
perceiver_mod = []
for t in modalities_per_task:
    for m in t:
        perceiver_mod.append(all_modalities[m])

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
model.to_logitslist = logits.to(device)

setting = '{}-{}-seed{}-bs{}-ep{}-enc_head{}-embd_dim{}-perceiver_dim{}-ttmax{}-embd_time{}-{}'.format(
    args.task,
    modeltype,
    args.seed,
    args.train_batch_size,
    args.num_train_epochs,
    args.num_heads,
    args.embed_dim,
    args.perceiver_dim,
    args.tt_max,
    args.embed_time,
    modalities_per_task
)
torch.save(model,'./checkpoints/mimic_iv.pt')
encoder = ModalityEncoders(args, device, modalities, args.tt_max, args.num_of_notes, BioBert)

records = train(
    model,
    all_train,
    all_valid,
    all_test,
    modalities_per_task,
    './checkpoints/mimic_iv.pt',
    args,
    encoder,
    setting,
    criterion=criterion,
    lr=0.0008,
    device=device,
    train_weights=[1.0, 1.0, 1.0],
    unsqueezing=[False, False, False],
    transpose=[False, False, False],
    weight_decay=0.001    
)
print('Experiment done!')