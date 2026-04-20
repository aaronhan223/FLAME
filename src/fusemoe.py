from typing import Iterable, Dict, List
from torch import Tensor
import torch
from torch import nn
import torch.nn.functional as F
import sys
import math
from src.module import *
from src.interp import *
import copy
import pdb

def findmodalityandindex(ms, mn):
    for i, m in enumerate(ms):
        if mn == m.name:
            return m, i
        elif mn.split('_')[0]==m.name.split('_')[0]:
            return m, i
    raise ValueError(f"modality {mn} not found in defined modalities")

class MULTCrossModel(nn.Module):
    def __init__(self,args,device,modalities,modalities_per_task,num_classes=None,latent_dim=512,modeltype=None,orig_d_ts=None,orig_reg_d_ts=None,orig_d_txt=None,ts_seq_num=None,text_seq_num=None, Biobert=None):
        """
        Construct a MulT Cross model.
        """
        super(MULTCrossModel, self).__init__()
        if modeltype!=None:
            self.modeltype=modeltype
        else:
            self.modeltype=args.modeltype
        self.num_heads = args.num_heads
        self.args = args
        self.layers = args.layers
        self.device=device
        self.kernel_size=args.kernel_size
        self.dropout=args.dropout
        self.attn_mask = False
        self.irregular_learn_emb_ts=args.irregular_learn_emb_ts
        self.irregular_learn_emb_text=args.irregular_learn_emb_text
        self.irregular_learn_emb_cxr=args.irregular_learn_emb_cxr
        self.irregular_learn_emb_ecg=args.irregular_learn_emb_ecg
        self.reg_ts=args.reg_ts
        self.TS_mixup=args.TS_mixup
        self.mixup_level=args.mixup_level
        self.task=args.task
        self.tt_max=args.tt_max
        self.tt_max_eicu=args.tt_max_eicu
        self.cross_method=args.cross_method
        self.modalities = modalities
        self.modalities_per_task = modalities_per_task
        self.num_modalities = len(self.modalities)
        self.use_pt_text_embeddings = args.use_pt_text_embeddings
        # self.token_type_embeddings = nn.Embedding(self.num_modalities, args.embed_dim)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(latent_dim*2),
            nn.Linear(latent_dim*2, num_classes)
        )
        self.d_ts = args.embed_dim
        self.d_txt = args.embed_dim
        self.d_cxr = args.embed_dim
        self.d_ecg = args.embed_dim
        self.d_t1 = args.embed_dim
        self.d_t2 = args.embed_dim
        self.d_t3 = args.embed_dim
        self.d_t4 = args.embed_dim
        self.d_t5 = args.embed_dim
        self.d_eicu = args.embed_dim
        self.d_cc = args.embed_dim
        self.d_mlo = args.embed_dim
        self.d_2dcc = args.embed_dim
        self.d_2dmlo = args.embed_dim
        self.d_allviews = args.embed_dim*4

        # if self.irregular_learn_emb_ts or self.irregular_learn_emb_text:
        #     self.time_query=torch.linspace(0, 1., self.tt_max)
        #     self.periodic = nn.Linear(1, args.embed_time - 1)
        #     self.linear = nn.Linear(1, 1)

        # if "TS" in self.modeltype:
        #     self.orig_d_ts=orig_d_ts
        #     self.d_ts=args.embed_dim
        #     self.ts_seq_num=ts_seq_num

        #     if self.irregular_learn_emb_ts:
        #         self.time_attn_ts=multiTimeAttention(self.orig_d_ts*2, self.d_ts, args.embed_time, 8)
 
        #     if self.reg_ts:
        #         self.orig_reg_d_ts=orig_reg_d_ts
        #         self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        #     if self.TS_mixup:
        #         if self.mixup_level=='batch':
        #             self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=args.dropout)
        #         elif self.mixup_level=='batch_seq':
        #             self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=args.dropout)
        #         elif self.mixup_level=='batch_seq_feature':
        #             self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=self.d_ts,dropout=args.dropout)
        #         else:
        #             raise ValueError("Unknown mixedup type")

        # if "Text" in self.modeltype:
        #     self.orig_d_txt = orig_d_txt
        #     self.d_txt = args.embed_dim
        #     self.text_seq_num = text_seq_num
        #     self.bertrep = BertForRepresentation(args, Biobert)

        #     if self.irregular_learn_emb_text:
        #         self.time_attn_text = multiTimeAttention(768, self.d_txt, args.embed_time, 8)
        #     else:
        #         self.proj_txt = nn.Conv1d(self.orig_d_txt, self.d_txt, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        # # if self.modeltype == "TS_CXR":
        # if "CXR" in self.modeltype:
        #     self.orig_d_cxr = 1024
        #     self.d_cxr = args.embed_dim
        #     self.cxr_seq_num = 5

        #     if self.irregular_learn_emb_cxr:
        #         self.time_attn_cxr = multiTimeAttention(1024, self.d_cxr, args.embed_time, 8)
        #     else:
        #         self.proj_cxr = nn.Conv1d(self.orig_d_cxr, self.d_cxr, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        # if "ECG" in self.modeltype:
        #     self.orig_d_ecg = 256
        #     self.d_ecg = args.embed_dim
        #     self.ecg_seq_num = 5

        #     if self.irregular_learn_emb_ecg:
        #         self.time_attn_ecg = multiTimeAttention(256, self.d_ecg, args.embed_time, 8)
        #     else:
        #         self.proj_ecg = nn.Conv1d(self.orig_d_ecg, self.d_ecg, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        output_dim = num_classes # args.num_labels
        # if self.modeltype=="TS_Text":
        if self.cross_method in ["self_cross", "moe", "hme", "flexmoe"]:
            self.trans_self_cross_ts_txt = self.get_cross_network(args, layers=args.cross_layers)
            self.proj1 = nn.ModuleDict()
            self.proj2 = nn.ModuleDict()
            self.out_layer = nn.ModuleDict()
            for t in self.modalities_per_task:
                dim = 0
                task = t[0].split('_')[1]   
                if f"TS_{task}" in t:
                    dim += self.d_ts
                if f"Text_{task}" in t:
                    dim += self.d_txt
                if f"CXR_{task}" in t:
                    dim += self.d_cxr
                if f"ECG_{task}" in t:
                    dim += self.d_ecg   
                if f"T1_{task}" in t:
                    dim += self.d_t1
                if f"T2_{task}" in t:
                    dim += self.d_t2
                if f"T3_{task}" in t:
                    dim += self.d_t3
                if f"T4_{task}" in t:
                    dim += self.d_t4
                if f"T5_{task}" in t:
                    dim += self.d_t5     
                if f"eicu_{task}" in t:
                    dim += self.d_eicu    
                if f"cc_{task}" in t:
                    dim += self.d_cc
                if f"mlo_{task}" in t:
                    dim += self.d_mlo
                if f"2dcc_{task}" in t:
                    dim += self.d_2dcc
                if f"2dmlo_{task}" in t:
                    dim += self.d_2dmlo
                if f"allviews_{task}" in t:
                    dim += self.d_allviews
                
                self.proj1[task] = nn.Linear(dim, dim)
                self.proj2[task] = nn.Linear(dim, dim)
                self.out_layer[task] = nn.Linear(dim, output_dim)
        else:
            # baseline fusion methods
            self.d_txt = args.embed_dim
            self.trans_ts_mem = self.get_network(self_type='ts_mem', layers=args.layers)
            self.trans_txt_mem = self.get_network(self_type='txt_mem', layers=args.layers)

            if self.cross_method=="MulT":
                self.trans_txt_with_ts=self.get_network(self_type='txt_with_ts',layers=args.cross_layers)
                self.trans_ts_with_txt=self.get_network(self_type='ts_with_txt',layers=args.cross_layers)
                self.proj1 = nn.Linear((self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
                self.proj2 = nn.Linear((self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
                self.out_layer = nn.Linear((self.d_ts+self.d_txt), output_dim)
            elif self.cross_method=="MAGGate":
                self.gate_fusion=MAGGate(inp1_size=self.d_txt, inp2_size=self.d_ts, dropout=self.dropout)
                self.proj1 = nn.Linear(self.d_txt, self.d_txt)
                self.proj2 = nn.Linear(self.d_txt, self.d_txt)
                self.out_layer = nn.Linear(self.d_txt, output_dim)
            elif self.cross_method=="Outer":
                self.outer_fusion=Outer(inp1_size=self.d_txt, inp2_size=self.d_ts)
                self.proj1 = nn.Linear(self.d_txt, self.d_txt)
                self.proj2 = nn.Linear(self.d_txt, self.d_txt)
                self.out_layer = nn.Linear(self.d_txt, output_dim)
            else:
                self.proj1 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                self.proj2 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                self.out_layer = nn.Linear(self.d_ts+self.d_txt, output_dim)
        
        # TODO: add baseline fusion methods for TS_CXR
        # if self.modeltype == "TS_CXR":
        #     if self.cross_method in ["self_cross", "moe", "moe_cross"]:
        #         self.trans_self_cross_ts_txt=self.get_cross_network(args, layers=args.cross_layers)
        #         self.proj1 = nn.Linear(self.d_ts+self.d_cxr, self.d_ts+self.d_cxr)
        #         self.proj2 = nn.Linear(self.d_ts+self.d_cxr, self.d_ts+self.d_cxr)
        #         self.out_layer = nn.Linear(self.d_ts+self.d_cxr, output_dim)

        # if 'ihm' in self.task or 'los' in self.task:
        #     self.loss_fct1=nn.CrossEntropyLoss()
        # elif 'pheno' in self.task:
        #     self.loss_fct1=nn.BCEWithLogitsLoss()
        # else:
        #     raise ValueError("Unknown task")

    def get_network(self, self_type='ts_mem', layers=-1):
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts:
                embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.tt_max, None
            else:
                embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.ts_seq_num, None
        elif self_type == 'txt_mem':
            if self.irregular_learn_emb_text:
                embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.tt_max, None
            else:
                embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.text_seq_num, None

        elif self_type =='txt_with_ts':
            if self.irregular_learn_emb_ts:
                embed_dim, q_seq_len,kv_seq_len = self.d_ts, self.tt_max, self.tt_max
            else:
                embed_dim, q_seq_len,kv_seq_len = self.d_ts, self.text_seq_num, self.ts_seq_num

        elif self_type =='ts_with_txt':
            if self.irregular_learn_emb_text:
                embed_dim, q_seq_len,kv_seq_len = self.d_txt, self.tt_max, self.tt_max
            else:
                embed_dim, q_seq_len,kv_seq_len = self.d_txt, self.ts_seq_num, self.text_seq_num
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  device=self.device,
                                  attn_dropout=self.dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask,
                                  q_seq_len=q_seq_len,
                                  kv_seq_len=kv_seq_len)

    def get_cross_network(self, args, layers=-1):
        embed_dim, q_seq_len = self.d_ts, self.tt_max
        return TransformerCrossEncoder(args=args,
                                        embed_dim=embed_dim,
                                        num_heads=self.num_heads,
                                        layers=layers,
                                        device=self.device,
                                        attn_dropout=self.dropout,
                                        relu_dropout=self.dropout,
                                        res_dropout=self.dropout,
                                        embed_dropout=self.dropout,
                                        attn_mask=self.attn_mask,
                                        q_seq_len_1=q_seq_len,
                                        modalities=self.modalities,
                                        num_modalities=self.num_modalities)

    def learn_time_embedding(self, tt):
        '''
        Time2Vec Module
        '''
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        # only two dimension?
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def _missing_indices(self, missing_idx):
        all_indices = torch.arange(len(missing_idx))
        missing_indices = torch.nonzero(missing_idx).squeeze(1)
        missing_mask = torch.ones(len(missing_idx), dtype=torch.bool)
        missing_mask[missing_indices] = False
        non_missing = all_indices[missing_mask]
        return missing_indices, non_missing

    # def forward(self, x_ts, x_ts_mask, ts_tt_list, cxr_missing=None, text_missing=None, ecg_missing=None, input_ids_sequences=None,
    #             attn_mask_sequences=None, text_emb=None, note_time_list=None, note_time_mask_list=None,
    #             labels=None, reg_ts=None, cxr_feats=None, cxr_time=None, cxr_time_mask=None, ecg_feats=None,
    #             ecg_time=None, ecg_time_mask=None):
    #     """
    #     dimension [batch_size, seq_len, n_features]

    #     """

        # if "TS" in self.modeltype:
        #     # mTAND module part
        #     if self.irregular_learn_emb_ts:
        #         time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
        #         time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

        #         x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
        #         x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)
        #         proj_x_ts_irg=self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask)
        #         proj_x_ts_irg=proj_x_ts_irg.transpose(0, 1)

        #     if self.reg_ts and reg_ts != None:
        #         x_ts_reg = reg_ts.transpose(1, 2)
        #         proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(x_ts_reg)
        #         # print('proj_x_ts_reg', torch.isnan(proj_x_ts_reg).any())
        #         proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

        #     if self.TS_mixup:
        #         if self.mixup_level=='batch':
        #             g_irg=torch.max(proj_x_ts_irg, dim=0).values
        #             g_reg =torch.max(proj_x_ts_reg, dim=0).values
        #             moe_gate=torch.cat([g_irg, g_reg], dim=-1)
        #         elif self.mixup_level=='batch_seq' or  self.mixup_level=='batch_seq_feature':
        #             moe_gate=torch.cat([proj_x_ts_irg,proj_x_ts_reg],dim=-1)
        #         else:
        #             raise ValueError("Unknown mixedup type")
        #         # print('moe_gate', torch.isnan(moe_gate).any())
        #         mixup_rate = self.moe(moe_gate)
        #         # print('mixup_rate', torch.isnan(mixup_rate).any())
        #         proj_x_ts = mixup_rate * proj_x_ts_irg + (1 - mixup_rate) * proj_x_ts_reg

        #     else:
        #         if self.irregular_learn_emb_ts:
        #             proj_x_ts=proj_x_ts_irg
        #         elif self.reg_ts:
        #             proj_x_ts=proj_x_ts_reg
        #         else:
        #             raise ValueError("Unknown time series type")
        #     proj_x_ts += self.token_type_embeddings(torch.zeros((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))

        # mod_count = 1
        # if "Text" in self.modeltype:
        #     # compute irregular clinical notes attention
        #     # if text_missing is None or torch.all(text_missing == 0):
        #     if self.use_pt_text_embeddings:
        #         x_txt = text_emb
        #     else:
        #         x_txt = self.bertrep(input_ids_sequences, attn_mask_sequences)

        #     if self.irregular_learn_emb_text:
        #         time_key = self.learn_time_embedding(note_time_list).to(self.device)
        #         if not self.irregular_learn_emb_ts:
        #             time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
        #         proj_x_txt=self.time_attn_text(time_query, time_key, x_txt, note_time_mask_list)
        #         proj_x_txt=proj_x_txt.transpose(0, 1)
        #     else:
        #         x_txt = x_txt.transpose(1, 2)
        #         proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)
        #         proj_x_txt = proj_x_txt.permute(2, 0, 1)
        #     if text_missing is None or torch.all(text_missing == 0):
        #         proj_x_txt += self.token_type_embeddings(torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))
        #     elif not torch.all(text_missing == 0):
        #         missing_indices, non_missing = self._missing_indices(text_missing)
        #         proj_x_txt[:, non_missing, :] += self.token_type_embeddings(torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
        #         proj_x_txt[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_ts.device)
        #     mod_count += 1

        # if "CXR" in self.modeltype:
        #     # compute irregular clinical notes attention
        #     if self.irregular_learn_emb_cxr:
        #         time_key = self.learn_time_embedding(cxr_time).to(self.device)
        #         if not self.irregular_learn_emb_ts:
        #             time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

        #         proj_x_cxr=self.time_attn_cxr(time_query, time_key, cxr_feats, cxr_time_mask)
        #         proj_x_cxr=proj_x_cxr.transpose(0, 1)
        #     else:
        #         cxr_feats = cxr_feats.transpose(1, 2)
        #         proj_x_cxr = cxr_feats if self.orig_d_cxr == self.d_cxr else self.proj_cxr(cxr_feats)
        #         proj_x_cxr = proj_x_cxr.permute(2, 0, 1)
        #     if cxr_missing is None or torch.all(cxr_missing == 0):
        #         proj_x_cxr += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))
        #     elif not torch.all(cxr_missing == 0):
        #         # proj_x_cxr = None
        #         missing_indices, non_missing = self._missing_indices(cxr_missing)
        #         proj_x_cxr[:, non_missing, :] += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
        #         proj_x_cxr[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_ts.device)
        #     mod_count += 1

        # if "ECG" in self.modeltype:
        #     # compute irregular ECG attention
        #     if self.irregular_learn_emb_cxr:
        #         time_key = self.learn_time_embedding(ecg_time).to(self.device)
        #         if not self.irregular_learn_emb_ts:
        #             time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

        #         proj_x_ecg=self.time_attn_ecg(time_query, time_key, ecg_feats, ecg_time_mask)
        #         proj_x_ecg=proj_x_ecg.transpose(0, 1)
        #     else:
        #         ecg_feats = ecg_feats.transpose(1, 2)
        #         proj_x_ecg = ecg_feats if self.orig_d_ecg == self.d_ecg else self.proj_ecg(ecg_feats)
        #         proj_x_ecg = proj_x_ecg.permute(2, 0, 1)
            
        #     if ecg_missing is None or torch.all(ecg_missing == 0):
        #         proj_x_ecg += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))
        #     elif not torch.all(ecg_missing == 0):
        #         # proj_x_ecg = None
        #         missing_indices, non_missing = self._missing_indices(ecg_missing)
        #         proj_x_ecg[:, non_missing, :] += self.token_type_embeddings(torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
        #         proj_x_ecg[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_ts.device)
        #     mod_count += 1
    def forward(self, multi_modality_data: Dict[str, Tensor], task=None, mask=None, use_recon=False, get_latent=False, get_pre_logits=False, latents=None, source_mode=None, get_catted=False, unimodal=False, null_pvi=False, labels=None):
        """
        :param data: a dictionary where keys are modality names and Tensor contain a batch
        of modality input data.
        :param mask:
        :return:
        """
        batch_sizes = set()
        latentout = []
        num_modalities = self.num_modalities
        modalities = []
        
        for m in multi_modality_data.keys():
            if 'TS' in m:
                proj_x_ts = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('TS')
                else:
                    modalities = ['TS']
            if 'Text' in m:
                proj_x_txt = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('Text')
                else:
                    modalities = ['Text']
            if 'CXR' in m:
                proj_x_cxr = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('CXR')
                else:
                    modalities = ['CXR']
            if 'ECG' in m:
                proj_x_ecg = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('ECG')
                else:
                    modalities = ['ECG']
            if 'T1' in m:
                proj_x_t1 = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('T1')
                else:
                    modalities = ['T1']
            if 'T2' in m:
                proj_x_t2 = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('T2')
                else:
                    modalities = ['T2']
            if 'T3' in m:
                proj_x_t3 = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('T3')
                else:
                    modalities = ['T3']
            if 'T4' in m:   
                proj_x_t4 = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('T4')
                else:
                    modalities = ['T4']
            if 'T5' in m:
                proj_x_t5 = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('T5')
                else:
                    modalities = ['T5']
            if 'eicu' in m:
                proj_x_eicu = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('eicu')
                else:
                    modalities = ['eicu']
            if f'cc_{task.upper()}' == m:
                proj_x_cc = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('cc')
                else:
                    modalities = ['cc']
            if f'mlo_{task.upper()}' == m:
                proj_x_mlo = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('mlo')
                else:
                    modalities = ['mlo']
            if f'2dcc_{task.upper()}' == m:
                proj_x_2dcc = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('2dcc')
                else:
                    modalities = ['2dcc']
            if f'2dmlo_{task.upper()}' == m:
                proj_x_2dmlo = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('2dmlo')
                else:
                    modalities = ['2dmlo']
            if f'allviews_{task.upper()}' == m:
                proj_x_allviews = multi_modality_data[m].transpose(1,0)
                if modalities:
                    modalities.append('allviews')
                else:
                    modalities = ['allviews']
        
        modalities = sorted(modalities)
        modalities = '_'.join(modalities)
        
        balance_loss = None
        
        if self.cross_method in ["self_cross", "moe", "hme", "flexmoe"]:
            # Maps each modality token → (local variable name, short name for the model)
            _proj_var = {
                'TS':   'proj_x_ts',   'Text': 'proj_x_txt',  'CXR':  'proj_x_cxr',
                'ECG':  'proj_x_ecg',  'T1':   'proj_x_t1',   'T2':   'proj_x_t2',
                'T3':   'proj_x_t3',   'T4':   'proj_x_t4',   'T5':   'proj_x_t5',
                'eicu': 'proj_x_eicu',  'cc': 'proj_x_cc',    'mlo': 'proj_x_mlo',
                '2dcc': 'proj_x_2dcc',    '2dmlo': 'proj_x_2dmlo', 'allviews': 'proj_x_allviews'
            }
            _short_name = {
                'TS':   'ts',    'Text': 'text',  'CXR':  'cxr',
                'ECG':  'ecg',   'T1':   't1',    'T2':   't2',
                'T3':   't3',    'T4':   't4',    'T5':   't5',
                'eicu': 'eicu',  'cc': 'cc',    'mlo': 'mlo',
                '2dcc': '2dcc',    '2dmlo': '2dmlo', 'allviews': 'allviews'
            }
            _lv = locals()
            tokens = modalities.split('_')
            proj_list = [_lv[_proj_var[t]] for t in tokens]
            name_list = [_short_name[t] for t in tokens]
            hiddens, balance_loss = self.trans_self_cross_ts_txt(proj_list, name_list)
            if hiddens is None:
                return None
            # h_txt_with_ts, h_ts_with_txt=hiddens
            last_hs = torch.cat([hid[-1] for hid in hiddens], dim=1)
            # last_hs = torch.cat([h_txt_with_ts[-1], h_ts_with_txt[-1]], dim=1)

        else:
            if 'CXR' in self.modeltype:
                proj_x_txt = proj_x_cxr
            if self.cross_method=="MulT":
                # ts --> txt
                h_txt_with_ts = self.trans_txt_with_ts(proj_x_txt, proj_x_ts, proj_x_ts)
                # txt --> ts
                h_ts_with_txt = self.trans_ts_with_txt(proj_x_ts, proj_x_txt, proj_x_txt)
                proj_x_ts = self.trans_ts_mem(h_txt_with_ts)
                proj_x_txt = self.trans_txt_mem(h_ts_with_txt)

                last_h_ts=proj_x_ts[-1]
                last_h_txt=proj_x_txt[-1]
                last_hs = torch.cat([last_h_ts,last_h_txt], dim=1)

            else:
                proj_x_ts = self.trans_ts_mem(proj_x_ts)
                proj_x_txt = self.trans_txt_mem(proj_x_txt)
                if self.cross_method=="MAGGate":
                    last_hs=self.gate_fusion(proj_x_txt[-1],proj_x_ts[-1])
                elif self.cross_method=="Outer":
                    last_hs=self.outer_fusion(proj_x_txt[-1],proj_x_ts[-1])
                else:
                    last_hs = torch.cat([proj_x_txt[-1],proj_x_ts[-1]], dim=1)
        
        # last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        
        if isinstance(self.proj1, nn.ModuleDict):
            last_hs_proj = F.dropout(F.relu(self.proj1[task](last_hs)), p=self.dropout, training=self.training)
        else:
            last_hs_proj = F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training)
        if isinstance(self.proj2, nn.ModuleDict):
            last_hs_proj = self.proj2[task](last_hs_proj)
        else:
            last_hs_proj = self.proj2(last_hs_proj)
        
        last_hs_proj += last_hs
        if get_pre_logits or get_latent or get_catted:
            return last_hs_proj
        
        if use_recon:
            return self.to_logits(last_hs_proj), None, balance_loss
        return self.to_logits(last_hs_proj), balance_loss
        # output = self.out_layer(last_hs_proj)

        # if 'ihm' in self.task or 'los' in self.task:
        #     if labels!=None:
        #         task_loss = self.loss_fct1(output, labels)
        #         return task_loss, balance_loss
        #     return torch.nn.functional.softmax(output,dim=-1)[:,1]

        # elif 'pheno' in self.task:
        #     if labels!=None:
        #         labels=labels.float()
        #         task_loss = self.loss_fct1(output, labels)
        #         return task_loss, balance_loss
        #     return torch.nn.functional.sigmoid(output)
