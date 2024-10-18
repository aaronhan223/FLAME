import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class multiTimeAttention(nn.Module):
    "mTAND module"
    def __init__(self, input_dim, device, nhidden=16,
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.device = device
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)]).to(self.device)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"

        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            if len(mask.shape)==3:
                mask=mask.unsqueeze(-1)
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -10000)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn=F.dropout(p_attn, p=dropout, training=self.training)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=0.1):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            mask = mask.to(self.device)
        value = value.unsqueeze(1)
        value = value.to(self.device)

        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)


class gateMLP(nn.Module):
    def __init__(self,input_dim,hidden_size,output_dim,dropout=0.1):
        super().__init__()

        self.gate = nn.Sequential(
             nn.Dropout(dropout),
             nn.Linear(input_dim, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size,output_dim),
             nn.Sigmoid()
        )
        self._initialize()

    def _initialize(self):
        for model in [self.gate]:
            for layer in model:
                if type(layer) in [nn.Linear]:
                    torch.nn.init.xavier_normal_(layer.weight)

    def forward(self,hidden_states ):
        gate_logits = self.gate(hidden_states)
        return gate_logits


class BertForRepresentation(nn.Module):
    """
    This class represents a BERT model for text representation.

    Args:
        args (object): The arguments for the model.
        BioBert (object): The BioBERT model.

    Attributes:
        bert (object): The BioBERT model.
        dropout (object): The dropout layer.
        model_name (str): The name of the model.
    """
    
    def __init__(self, args,BioBert):
        super().__init__()
        self.bert = BioBert

        self.dropout = torch.nn.Dropout(BioBert.config.hidden_dropout_prob)
        self.model_name=args.model_name

    def forward(self, input_ids_sequence, attention_mask_sequence, sent_idx_list=None , doc_idx_list=None):
        """
        Forward pass of the model.

        Args:
            input_ids_sequence (List[Tensor]): List of input token IDs for each sequence.
            attention_mask_sequence (List[Tensor]): List of attention masks for each sequence.
            sent_idx_list (List[int], optional): List of sentence indices. Defaults to None.
            doc_idx_list (List[int], optional): List of document indices. Defaults to None.

        Returns:
            Tensor: Text embeddings for each sequence.
        """
        txt_arr = []

        for input_ids, attention_mask in zip(input_ids_sequence, attention_mask_sequence):

            if 'Longformer' in self.model_name:

                attention_mask-=1

                text_embeddings=self.bert(input_ids, global_attention_mask=attention_mask)
            else:
                text_embeddings=self.bert(input_ids, attention_mask=attention_mask)
            text_embeddings= text_embeddings[0][:,0,:]
            text_embeddings = self.dropout(text_embeddings)
            txt_arr.append(text_embeddings)

        txt_arr=torch.stack(txt_arr)
        return txt_arr


class ModalityEncoders(nn.Module):
    def __init__(
        self,
        args,
        device,
        modalities,
        ts_seq_num,
        text_seq_num,
        Biobert,
        orig_d_ts=30,
        orig_reg_d_ts=60,
        orig_d_txt=768,
    ):
        super(ModalityEncoders, self).__init__()
        
        self.modalities = modalities
        self.args = args
        self.device = device
        self.reg_ts=args.reg_ts
        self.TS_mixup=args.TS_mixup
        self.tt_max=args.tt_max
        self.reg_ts=args.reg_ts
        self.kernel_size=args.kernel_size
        self.dropout=args.dropout
        self.use_pt_text_embeddings = args.use_pt_text_embeddings
        
        self.time_query=torch.linspace(0, 1., self.tt_max)
        self.periodic = nn.Linear(1, args.embed_time - 1).to(self.device)
        self.linear = nn.Linear(1, 1).to(self.device)

        if "TS" in self.modalities:
            self.orig_d_ts=orig_d_ts
            self.d_ts=args.embed_dim
            self.ts_seq_num=ts_seq_num
            self.time_attn_ts=multiTimeAttention(self.orig_d_ts*2, self.device, self.d_ts, args.embed_time, args.num_heads)
            
            if self.reg_ts:
                self.orig_reg_d_ts=orig_reg_d_ts
                self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False).to(self.device)
            if self.TS_mixup:
                self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=self.dropout).to(self.device)
        
        if "Text" in self.modalities:
            self.orig_d_txt = orig_d_txt
            self.text_seq_num = text_seq_num
            self.bertrep = BertForRepresentation(args, Biobert)
            self.time_attn_text = multiTimeAttention(self.orig_d_txt, self.device, args.embed_dim, args.embed_time, args.num_heads)

        if "CXR" in self.modalities:
            self.time_attn_cxr = multiTimeAttention(1024, self.device, args.embed_dim, args.embed_time, args.num_heads)

        if "ECG" in self.modalities:
            self.time_attn_ecg = multiTimeAttention(256, self.device, args.embed_dim, args.embed_time, args.num_heads)

    def learn_time_embedding(self, tt):
        '''
        Time2Vec Module
        '''
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x_ts, x_ts_mask, ts_tt_list, cxr_missing=None, text_missing=None, ecg_missing=None, input_ids_sequences=None,
                attn_mask_sequences=None, text_emb=None, note_time_list=None, note_time_mask_list=None,
                labels=None, reg_ts=None, cxr_feats=None, cxr_time=None, cxr_time_mask=None, ecg_feats=None,
                ecg_time=None, ecg_time_mask=None, modalities=None):
        
        embeddings = {}
        if any("TS" in s for s in modalities):
            idx = [index for index, mod in enumerate(modalities) if "TS" in mod][0]
            time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
            time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
            x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
            x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)
            proj_x_ts_irg=self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask)
            proj_x_ts_irg=proj_x_ts_irg.transpose(0, 1)
            
            if self.reg_ts and reg_ts != None:
                x_ts_reg = reg_ts.transpose(1, 2).to(self.device)
                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(x_ts_reg)
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

            g_irg=torch.max(proj_x_ts_irg, dim=0).values
            g_reg =torch.max(proj_x_ts_reg, dim=0).values
            moe_gate=torch.cat([g_irg, g_reg], dim=-1)
            mixup_rate = self.moe(moe_gate)
            proj_x_ts = mixup_rate * proj_x_ts_irg + (1 - mixup_rate) * proj_x_ts_reg
            embeddings[modalities[idx]] = proj_x_ts.transpose(0, 1)
        
        if any("Text" in s for s in modalities):
            idx = [index for index, mod in enumerate(modalities) if "Text" in mod][0]
            if self.use_pt_text_embeddings:
                x_txt = text_emb
            else:
                x_txt = self.bertrep(input_ids_sequences, attn_mask_sequences)
            time_key = self.learn_time_embedding(note_time_list).to(self.device)
            proj_x_txt = self.time_attn_text(time_query, time_key, x_txt, note_time_mask_list)
            embeddings[modalities[idx]] = proj_x_txt
        
        if any("CXR" in s for s in modalities):
            idx = [index for index, mod in enumerate(modalities) if "CXR" in mod][0]
            time_key = self.learn_time_embedding(cxr_time).to(self.device)
            proj_x_cxr = self.time_attn_cxr(time_query, time_key, cxr_feats, cxr_time_mask)
            embeddings[modalities[idx]] = proj_x_cxr
        
        if any("ECG" in s for s in modalities):
            idx = [index for index, mod in enumerate(modalities) if "ECG" in mod][0]
            time_key = self.learn_time_embedding(ecg_time).to(self.device)
            proj_x_ecg = self.time_attn_ecg(time_query, time_key, ecg_feats, ecg_time_mask)
            if torch.isnan(proj_x_ecg).any():
                embeddings[modalities[idx]] = torch.nan_to_num(proj_x_ecg, nan=0.0)
            else:
                embeddings[modalities[idx]] = proj_x_ecg

        return embeddings