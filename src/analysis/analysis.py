import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
from crossattnperceiver import MultiModalityPerceiver, InputModality
from train_structure_multitask_mimic import train
from encoders import ModalityEncoders, FSEncoder
from utils import create_directory, dump_pickle
from preprocess.preprocess_eicu import *
import torch
from accelerate import Accelerator
torch.multiprocessing.set_sharing_strategy('file_system')
from datasets.mimic.get_data_mimic_iv import data_prepare as prepare_mimic
from get_data_eicu import data_prepare as prepare_eicu

def create_model(args, device, logits):
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
    return model

def layerwise_rank_analysis(model, tol=1e-5):
    """
    Computes the numerical rank (low-rank structure) of each parameter tensor in the model.
    
    Args:
        model: torch.nn.Module
        tol: threshold for singular values to consider as non-zero (for numerical rank)
    
    Returns:
        dict mapping layer names -> rank info
    """
    rank_info = {}

    for name, param in model.named_parameters():
        if param.dim() >= 2:  # Only analyze weight matrices (not biases, batchnorm params, etc.)
            W = param.detach().cpu()
            # Flatten convolutional kernels: (out_channels, in_channels, kH, kW) → (out_channels, -1)
            if W.dim() > 2:
                W = W.view(W.size(0), -1)
            
            # Compute singular values via SVD
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            
            # Count significant singular values
            rank = torch.sum(S > tol).item()
            
            rank_info[name] = {
                "shape": tuple(param.shape),
                "rank": rank,
                "rank_ratio": rank / min(W.shape),  # normalized rank
            }

    return rank_info

import torch

def layerwise_concat_rank(model1, model2, tol=1e-5):
    """
    Computes numerical rank for concatenated parameters layer-by-layer
    between two models with the same architecture.
    """
    rank_info = {}
    
    for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2, f"Layer mismatch: {name1} vs {name2}"
        
        if p1.dim() >= 2:
            W1 = p1.detach().cpu()
            W2 = p2.detach().cpu()
            
            # Flatten convolutional kernels into matrices
            if W1.dim() > 2:
                W1 = W1.view(W1.size(0), -1)
                W2 = W2.view(W2.size(0), -1)
            
            # Concatenate along output dimension (rows)
            W_concat = torch.cat([W1, W2], dim=0)
            
            # Compute SVD
            S = torch.linalg.svdvals(W_concat)
            rank = torch.sum(S > tol).item()
            
            rank_info[name1] = {
                "shape": tuple(W1.shape),
                "concat_rank": rank,
                "concat_rank_ratio": rank / min(W_concat.shape),
            }

    return rank_info

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='mimic', choices=['mimic', 'eicu'])
parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--perceiver_dim', type=int, default=64)
parser.add_argument("--ihm_mod", type=str, default='TS-Text-CXR', help="Modality compoenents for IHM task.")
parser.add_argument("--los_mod", type=str, default='TS-Text-CXR', help="Modality compoenents for LOS task.")
parser.add_argument("--pheno_mod", type=str, default='', help="Modality compoenents for PHENO task.")
parser.add_argument("--rad_mod", type=str, default='', help="Modality compoenents for readmission task.")
parser.add_argument("--mor_mod", type=str, default='', help="Modality compoenents for mortality task.")
args = parser.parse_args()

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
all_modalities['T1_MOR'] = InputModality(
    name='T1_MOR',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['T2_MOR'] = InputModality(
    name='T2_MOR',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['T3_MOR'] = InputModality(
    name='T3_MOR',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['T4_MOR'] = InputModality(
    name='T4_MOR',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['T5_MOR'] = InputModality(
    name='T5_MOR',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['T1_RAD'] = InputModality(
    name='T1_RAD',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['T2_RAD'] = InputModality(
    name='T2_RAD',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['T3_RAD'] = InputModality(
    name='T3_RAD',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['T4_RAD'] = InputModality(
    name='T4_RAD',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
all_modalities['T5_RAD'] = InputModality(
    name='T5_RAD',
    input_channels=args.embed_dim,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
# TODO: each feature a modality? clustering feature?
# TODO: should we keep feature specific encoders?
# import pdb; pdb.set_trace()

modalities_per_task = []
logits = torch.nn.ModuleList()
ihm_mods = list(map(lambda s: s + '_IHM', args.ihm_mod.split("-")))
assert len(ihm_mods) > 1, "At least two modalities per task!"
modalities_per_task.append(ihm_mods)
logit_dim = len(ihm_mods) * (len(ihm_mods) - 1) * args.perceiver_dim
logits.append(torch.nn.Sequential(torch.nn.LayerNorm(logit_dim), torch.nn.Linear(logit_dim, 2)))

perceiver_mod = []
for t in modalities_per_task:
    for m in t:
        perceiver_mod.append(all_modalities[m])

# model1 = create_model(args, 'cuda', logits=torch.nn.Linear(args.perceiver_dim, 1))
model1 = torch.load('/cis/home/schaud35/clinical-highmmt/src/checkpoints/mimic_iv_ihm_TS-Text-CXR.pt')

modalities_per_task = []
logits = torch.nn.ModuleList()
los_mods = list(map(lambda s: s + '_LOS', args.los_mod.split("-")))
assert len(los_mods) > 1, "At least two modalities per task!"
modalities_per_task.append(los_mods)
logit_dim = len(los_mods) * (len(los_mods) - 1) * args.perceiver_dim
logits.append(torch.nn.Sequential(torch.nn.LayerNorm(logit_dim), torch.nn.Linear(logit_dim, 2)))

perceiver_mod = []
for t in modalities_per_task:
    for m in t:
        perceiver_mod.append(all_modalities[m])

# model2 = create_model(args, 'cuda', logits=torch.nn.Linear(args.perceiver_dim, 1))
model2 = torch.load('/cis/home/schaud35/clinical-highmmt/src/checkpoints/mimic_iv_los_TS-Text-CXR.pt')

# Option 2: Check if the model architectures are identical
same_architecture = str(model1) == str(model2)
print("Same architecture:", same_architecture)

# Option 3: Compare layer names and parameters
def same_model_structure(m1, m2):
    return all(a == b for a, b in zip(m1.state_dict().keys(), m2.state_dict().keys()))

print("Same structure:", same_model_structure(model1, model2))

ranks = layerwise_rank_analysis(model1, tol=1e-4)

for name, info in ranks.items():
    print(f"{name}: rank = {info['rank']} / {min(info['shape'])} ({info['rank_ratio']:.2f})")

ranks = layerwise_rank_analysis(model2, tol=1e-4)

for name, info in ranks.items():
    print(f"{name}: rank = {info['rank']} / {min(info['shape'])} ({info['rank_ratio']:.2f})")

ranks = layerwise_concat_rank(model1, model2, tol=1e-4)
for name, info in ranks.items():
    print(f"{name}: concat rank = {info['concat_rank']} ({info['concat_rank_ratio']:.2f})")

import pdb; pdb.set_trace()