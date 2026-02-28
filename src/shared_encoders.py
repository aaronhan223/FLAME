import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layer import RelTemporalEncoding, TransformerBlock
# from utils import count_parameters
import pdb
from src.encoders import *

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layer import RelTemporalEncoding, TransformerBlock
# from utils import count_parameters
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

class TimeQueryEncoder(nn.Module):
    def __init__(self, tt_max, embed_time, device):
        super().__init__()

        self.device = device
        self.tt_max = tt_max

        # fixed query positions
        self.register_buffer(
            "time_query",
            torch.linspace(0, 1., tt_max)
        )

        # Time2Vec layers
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

    def forward(self, tt=None):
        if tt is None:
            tt = self.time_query.unsqueeze(0).unsqueeze(-1)  # [1, tt_max, 1]
        else:
            tt = tt.to(self.device).unsqueeze(-1)  # [batch_size, seq_len, 1]

        out1 = self.linear(tt)
        out2 = torch.sin(self.periodic(tt))

        return torch.cat([out1, out2], dim=-1)


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
        shared_time_encoder=None
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
        
        # self.time_query=torch.linspace(0, 1., self.tt_max)
        self.shared_time_encoder = shared_time_encoder
        # self.periodic = nn.Linear(1, args.embed_time - 1).to(self.device)
        # self.linear = nn.Linear(1, 1).to(self.device)

        if "TS" in self.modalities:
            self.orig_d_ts=orig_d_ts
            self.d_ts=args.embed_dim
            self.ts_seq_num=ts_seq_num
            self.time_attn_ts=multiTimeAttention(self.orig_d_ts*2, self.device, self.d_ts, args.embed_time, args.num_heads)
            
            if self.reg_ts:
                self.orig_reg_d_ts=orig_reg_d_ts
                self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False).to(self.device)
            if self.TS_mixup:
                self.moe = gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=self.dropout).to(self.device)
        
        if "Text" in self.modalities:
            self.orig_d_txt = orig_d_txt
            self.text_seq_num = text_seq_num
            self.bertrep = BertForRepresentation(args, Biobert).to(self.device)
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
        # time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
        time_query = self.shared_time_encoder()
        if any("TS" in s for s in modalities):
            idx = [index for index, mod in enumerate(modalities) if "TS" in mod][0]
            # time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
            time_key_ts = self.shared_time_encoder(ts_tt_list).to(self.device)
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
                x_txt = self.bertrep(input_ids_sequences.to(self.device), attn_mask_sequences.to(self.device))
               
            # time_key = self.learn_time_embedding(note_time_list).to(self.device)
            time_key = self.shared_time_encoder(note_time_list).to(self.device)
            
            proj_x_txt = self.time_attn_text(time_query, time_key, x_txt, note_time_mask_list)
            embeddings[modalities[idx]] = proj_x_txt
        
        if any("CXR" in s for s in modalities):
            idx = [index for index, mod in enumerate(modalities) if "CXR" in mod][0]
            # time_key = self.learn_time_embedding(cxr_time).to(self.device)
            time_key = self.shared_time_encoder(cxr_time).to(self.device)
            proj_x_cxr = self.time_attn_cxr(time_query, time_key, cxr_feats, cxr_time_mask)
            embeddings[modalities[idx]] = proj_x_cxr
        
        if any("ECG" in s for s in modalities):
            idx = [index for index, mod in enumerate(modalities) if "ECG" in mod][0]
            # time_key = self.learn_time_embedding(ecg_time).to(self.device)
            time_key = self.shared_time_encoder(ecg_time).to(self.device)
            proj_x_ecg = self.time_attn_ecg(time_query, time_key, ecg_feats, ecg_time_mask)
            if torch.isnan(proj_x_ecg).any():
                embeddings[modalities[idx]] = torch.nan_to_num(proj_x_ecg, nan=0.0)
            else:
                embeddings[modalities[idx]] = proj_x_ecg
        return embeddings


def batch_to_multihot(x, num_labels):
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    multihot_x = torch.zeros(batch_size, seq_len, num_labels).to(x.device)
    multihot_x[torch.arange(batch_size).unsqueeze(1).repeat(1, seq_len),
    torch.arange(seq_len).repeat(batch_size, 1), x] = 1
    return multihot_x


class FSEncoder(nn.Module):
    def __init__(
            self,
            tokenizer,
            embedding_size: int,
            pretrained_embedding: bool,
            dropout: float,
            layers: int,
            heads: int,
            hidden_size: int,
            device="cpu",
            shared_time_encoder=None,
            args=None
    ):
        super(FSEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self.pretrained_embedding = pretrained_embedding
        self.dropout = dropout
        self.layers = layers
        self.heads = heads
        self.hidden_size = hidden_size
        self.device = device
        self.shared_time_encoder = shared_time_encoder
        self.tt_max = args.tt_max
        # embedding
        # if pretrained_embedding:
        #     mm = torch.Tensor(self.tokenizer.code_embeddings)
        #     assert self.tokenizer.code_vocabs_size == mm.shape[0]
        #     self.code_embedding = nn.Sequential(
        #         nn.Embedding(self.tokenizer.code_vocabs_size, mm.shape[1]),
        #         nn.Linear(mm.shape[1], embedding_size)
        #     ).to(self.device)
        #     self.code_embedding[0].weight.data.copy_(mm)
        #     self.code_embedding[0].weight.requires_grad = False
        # else:
        #     self.code_embedding = nn.Embedding(self.tokenizer.code_vocabs_size, embedding_size, padding_idx=0).to(self.device)
        self.code_embedding = nn.Embedding(self.tokenizer.code_vocabs_size, embedding_size, padding_idx=0).to(self.device)
        self.type_embedding = nn.Embedding(self.tokenizer.type_vocabs_size, embedding_size, padding_idx=0).to(self.device)
        self.timestamp_embedding = RelTemporalEncoding(embedding_size).to(self.device)
        self.age_embedding = nn.Embedding(self.tokenizer.age_vocabs_size, embedding_size, padding_idx=0).to(self.device)
        self.gender_embedding = nn.Embedding(self.tokenizer.gender_vocabs_size, embedding_size, padding_idx=0).to(self.device)
        self.ethnicity_embedding = nn.Embedding(self.tokenizer.ethnicity_vocabs_size, embedding_size, padding_idx=0).to(self.device)

        # encoder
        self.transformer = nn.ModuleList([TransformerBlock(embedding_size, heads, dropout) for _ in range(layers)]).to(self.device)
        self.dropout = nn.Dropout(dropout)
        self.projector = nn.Linear(embedding_size, hidden_size).to(self.device)

        # aggregator
        self.num_event_types = self.tokenizer.type_vocabs_size
        self.num_fake_event_types = len(self.tokenizer.type_vocabs.init_words)

        # self.time_attn_fs = multiTimeAttention(
        #     input_dim=self.hidden_size,
        #     device=self.device,
        #     nhidden=self.hidden_size,
        #     embed_time=args.embed_time,
        #     num_heads=args.num_heads
        # )
        self.time_attn_fs = multiTimeAttention(128, self.device, args.embed_dim, args.embed_time, args.num_heads)
        # print(f"# of params: {count_parameters(self)}")

    @property
    def num_types(self):
        return self.num_event_types - self.num_fake_event_types + 1

    def forward(
            self, codes, types, timestamps, ages, genders, ethnicities, modalities
    ):
        
        
        codes = codes.to(self.device)
        types = types.to(self.device)
        timestamps = timestamps.to(self.device)
        ages = ages.to(self.device)
        genders = genders.to(self.device)
        ethnicities = ethnicities.to(self.device)

        mask = (types != 0)
        mask = torch.einsum("ab,ac->abc", mask, mask)
        """ embedding """
        # [# admissions, # batch_codes, embedding_size]
        codes_emb = self.code_embedding(codes)
        types_emb = self.type_embedding(types)
        timestamps_emb = self.timestamp_embedding(timestamps) / math.sqrt(self.embedding_size)
        ages_emb = self.age_embedding(ages)
        genders_emb = self.gender_embedding(genders)
        ethnicities_emb = self.ethnicity_embedding(ethnicities)
        demographics_emb = (ages_emb + genders_emb + ethnicities_emb).unsqueeze(1)
        emb = codes_emb + types_emb + timestamps_emb + demographics_emb
        # sum every embedding
        emb[mask.sum(dim=-1) == 0] = 0

        """ transformer """
        x = emb
        for transformer in self.transformer:
            x = transformer(x, mask)  # [# admissions, # batch_codes, embedding_size]
        x = self.dropout(x)
        x = self.projector(x)  # [# admissions, # batch_codes, hidden_size]
        
        time_query = self.shared_time_encoder()              # [1, tt_max, embed_time]
        time_key   = self.shared_time_encoder((timestamps.float() / self.tt_max).clamp(0.00, 1.0))    # [B, seq, embed_time]

        # Global mask (all valid tokens)
        global_time_mask = (types != 0)  # [B, seq]

        # SINGLE attention call
        # x_time = self.time_attn_fs(
        #     time_query,
        #     time_key,
        #     x,
        #     global_time_mask
        # )  # [B, tt_max, hidden]
        
        # x_time_mask = (types != 0)                           # [B, seq]

        # x = self.time_attn_fs(time_query, time_key, x, x_time_mask)

        # idx = [index for index, mod in enumerate(modalities) if "eicu" in mod][0]
        # embedding = {modalities[idx]: x}
        
        # return embedding
        cls_emb = x[:, 0, :]

        """ aggregate """
        embedding = {}
        # loop over your event-type modalities (T1...T5)
        for i, mod in enumerate(modalities):

            # assume modality i corresponds to type_id = i+1 (adjust if needed)
            type_id = i + 1

            # mask tokens of this type
            type_mask = (types == type_id)                # [B, seq]

            # if no tokens exist, fallback to zeros
            if type_mask.sum().item() == 0:
                embedding[mod] = torch.zeros(
                    codes.size(0), self.tt_max, self.hidden_size
                ).to(self.device)
                continue
            
            # apply time attention using only this type's events
            proj_x_type = self.time_attn_fs(
                time_query,
                time_key,
                x,
                type_mask
            )  # [B, 48, D]
            # batch_has_type = type_mask.any(dim=1).float().unsqueeze(-1).unsqueeze(-1)
            # embedding[mod] = x_time * batch_has_type
            embedding[mod] = proj_x_type
        return embedding

        types_multihot = batch_to_multihot(types, self.num_event_types)
        mask = (types != 0)
        types_multihot[~mask] = 0

        group_count = torch.sum(types_multihot, dim=1)
        inv_group_count = nn.functional.normalize(types_multihot, p=1, dim=1)
        type_representations = torch.matmul(inv_group_count.transpose(1, 2), x)
        type_representations[:, 0, :] = cls_emb
        type_representations[group_count == 0] = torch.repeat_interleave(cls_emb, (group_count == 0).sum(dim=1), dim=0)
        type_representations = type_representations.permute(1, 0, 2)  # [# types, # admissions, hidden_size]
        select = [False] * self.num_fake_event_types + [True] * (self.num_event_types - self.num_fake_event_types)
        select[0] = True
        type_representations = type_representations[select]
        
        embedding = {}
        for i in range(type_representations.shape[0]):
            embedding[modalities[i]] = type_representations[i].unsqueeze(0).permute(1, 0, 2)
        return embedding

# class SharedModalityEncoders(nn.Module):
#     def __init__(
#         self,
#         args=None,
#         device="cpu",
#         modalities=None,
#         ts_seq_num=None,
#         text_seq_num=None,
#         Biobert=None,
#         orig_d_ts=30,
#         orig_reg_d_ts=60,
#         orig_d_txt=768,
#         eicutokenizer = None,
#         embedding_size: int = 128,
#         pretrained_embedding: bool = True,
#         dropout: float = 0.1,
#         layers: int = 2,
#         heads: int = 2,
#         hidden_size: int = 128
#     ):
#         super(SharedModalityEncoders, self).__init__()
#         # MIMIC

#         self.modalities = modalities
#         self.args = args
#         self.device = device
#         self.reg_ts=args.reg_ts
#         self.TS_mixup=args.TS_mixup
#         self.tt_max=args.tt_max
#         self.reg_ts=args.reg_ts
#         self.kernel_size=args.kernel_size
#         self.dropout=args.dropout
#         self.use_pt_text_embeddings = args.use_pt_text_embeddings

#         self.time_query=torch.linspace(0, 1., self.tt_max)
#         self.periodic = nn.Linear(1, args.embed_time - 1).to(self.device)
#         self.linear = nn.Linear(1, 1).to(self.device)

#         if "TS" in self.modalities:
#             self.orig_d_ts=orig_d_ts
#             self.d_ts=args.embed_dim
#             self.ts_seq_num=ts_seq_num
#             self.time_attn_ts=multiTimeAttention(self.orig_d_ts*2, self.device, self.d_ts, args.embed_time, args.num_heads)
            
#             if self.reg_ts:
#                 self.orig_reg_d_ts=orig_reg_d_ts
#                 self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False).to(self.device)
#             if self.TS_mixup:
#                 self.moe = gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=self.dropout).to(self.device)
        
#         if "Text" in self.modalities:
#             self.orig_d_txt = orig_d_txt
#             self.text_seq_num = text_seq_num
#             self.bertrep = BertForRepresentation(args, Biobert).to(self.device)
#             self.time_attn_text = multiTimeAttention(self.orig_d_txt, self.device, args.embed_dim, args.embed_time, args.num_heads)

#         if "CXR" in self.modalities:
#             self.time_attn_cxr = multiTimeAttention(1024, self.device, args.embed_dim, args.embed_time, args.num_heads)

#         if "ECG" in self.modalities:
#             self.time_attn_ecg = multiTimeAttention(256, self.device, args.embed_dim, args.embed_time, args.num_heads)

#         # eICU
#         self.tokenizer = eicutokenizer
#         self.embedding_size = embedding_size
#         self.pretrained_embedding = pretrained_embedding
#         self.dropout = dropout
#         self.layers = layers
#         self.heads = heads
#         self.hidden_size = hidden_size
#         self.device = device
#         # embedding
#         if pretrained_embedding:
#             mm = torch.Tensor(self.tokenizer.code_embeddings)
#             assert self.tokenizer.code_vocabs_size == mm.shape[0]
#             self.code_embedding = nn.Sequential(
#                 nn.Embedding(self.tokenizer.code_vocabs_size, mm.shape[1]),
#                 nn.Linear(mm.shape[1], embedding_size)
#             ).to(self.device)
#             self.code_embedding[0].weight.data.copy_(mm)
#             self.code_embedding[0].weight.requires_grad = False
#         else:
#             self.code_embedding = nn.Embedding(self.tokenizer.code_vocabs_size, embedding_size, padding_idx=0).to(self.device)
#         self.type_embedding = nn.Embedding(self.tokenizer.type_vocabs_size, embedding_size, padding_idx=0).to(self.device)
#         self.timestamp_embedding = RelTemporalEncoding(embedding_size).to(self.device)
#         self.age_embedding = nn.Embedding(self.tokenizer.age_vocabs_size, embedding_size, padding_idx=0).to(self.device)
#         self.gender_embedding = nn.Embedding(self.tokenizer.gender_vocabs_size, embedding_size, padding_idx=0).to(self.device)
#         self.ethnicity_embedding = nn.Embedding(self.tokenizer.ethnicity_vocabs_size, embedding_size, padding_idx=0).to(self.device)

#         # encoder
#         self.transformer = nn.ModuleList([TransformerBlock(embedding_size, heads, dropout) for _ in range(layers)]).to(self.device)
#         self.dropout = nn.Dropout(dropout)
#         self.projector = nn.Linear(embedding_size, hidden_size).to(self.device)

#         # aggregator
#         self.num_event_types = self.tokenizer.type_vocabs_size
#         self.num_fake_event_types = len(self.tokenizer.type_vocabs.init_words)

#     def learn_time_embedding(self, tt):
#         '''
#         Time2Vec Module
#         '''
#         tt = tt.to(self.device)
#         tt = tt.unsqueeze(-1)
#         out2 = torch.sin(self.periodic(tt))
#         out1 = self.linear(tt)
#         return torch.cat([out1, out2], -1)

#     @property
#     def num_types(self):
#         return self.num_event_types - self.num_fake_event_types + 1

#     def forward(self, x_ts=None, x_ts_mask=None, ts_tt_list=None, cxr_missing=None, text_missing=None, ecg_missing=None, input_ids_sequences=None,
#                 attn_mask_sequences=None, text_emb=None, note_time_list=None, note_time_mask_list=None,
#                 labels=None, reg_ts=None, cxr_feats=None, cxr_time=None, cxr_time_mask=None, ecg_feats=None,
#                 ecg_time=None, ecg_time_mask=None, modalities=None,
#                 codes=None, types=None, timestamps=None, ages=None, genders=None, ethnicities=None):
#         import pdb; pdb.set_trace()
#         embeddings = {}
#         time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
#         if any("TS" in s for s in modalities):
#             idx = [index for index, mod in enumerate(modalities) if "TS" in mod][0]
#             time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
#             x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
#             x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)
#             proj_x_ts_irg=self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask)
#             proj_x_ts_irg=proj_x_ts_irg.transpose(0, 1)
            
#             if self.reg_ts and reg_ts != None:
#                 x_ts_reg = reg_ts.transpose(1, 2).to(self.device)
#                 proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(x_ts_reg)
#                 proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

#             g_irg=torch.max(proj_x_ts_irg, dim=0).values
#             g_reg =torch.max(proj_x_ts_reg, dim=0).values
#             moe_gate=torch.cat([g_irg, g_reg], dim=-1)
#             mixup_rate = self.moe(moe_gate)
#             proj_x_ts = mixup_rate * proj_x_ts_irg + (1 - mixup_rate) * proj_x_ts_reg
#             embeddings[modalities[idx]] = proj_x_ts.transpose(0, 1)
        
#         if any("Text" in s for s in modalities):
#             idx = [index for index, mod in enumerate(modalities) if "Text" in mod][0]
#             if self.use_pt_text_embeddings:
#                 x_txt = text_emb
#             else:
#                 x_txt = self.bertrep(input_ids_sequences.to(self.device), attn_mask_sequences.to(self.device))
               
#             time_key = self.learn_time_embedding(note_time_list).to(self.device)
#             proj_x_txt = self.time_attn_text(time_query, time_key, x_txt, note_time_mask_list)
#             embeddings[modalities[idx]] = proj_x_txt
        
#         if any("CXR" in s for s in modalities):
#             idx = [index for index, mod in enumerate(modalities) if "CXR" in mod][0]
#             time_key = self.learn_time_embedding(cxr_time).to(self.device)
#             proj_x_cxr = self.time_attn_cxr(time_query, time_key, cxr_feats, cxr_time_mask)
#             embeddings[modalities[idx]] = proj_x_cxr
        
#         if any("ECG" in s for s in modalities):
#             idx = [index for index, mod in enumerate(modalities) if "ECG" in mod][0]
#             time_key = self.learn_time_embedding(ecg_time).to(self.device)
#             proj_x_ecg = self.time_attn_ecg(time_query, time_key, ecg_feats, ecg_time_mask)
#             if torch.isnan(proj_x_ecg).any():
#                 embeddings[modalities[idx]] = torch.nan_to_num(proj_x_ecg, nan=0.0)
#             else:
#                 embeddings[modalities[idx]] = proj_x_ecg

#         select = [False] * self.num_event_types
#         iseicu = False
#         eicu_idx = []
#         if any("T1" in s for s in modalities):
#             iseicu = True
#             eicu_idx.append([index for index, mod in enumerate(modalities) if "T1" in mod][0])
#             select[0] = True
#         if any("T2" in s for s in modalities):
#             iseicu = True
#             eicu_idx.append([index for index, mod in enumerate(modalities) if "T2" in mod][0])
#             select[self.num_fake_event_types+1] = True
#         if any("T3" in s for s in modalities):
#             iseicu = True
#             eicu_idx.append([index for index, mod in enumerate(modalities) if "T3" in mod][0])
#             select[self.num_fake_event_types+2] = True
#         if any("T4" in s for s in modalities):
#             iseicu = True
#             eicu_idx.append([index for index, mod in enumerate(modalities) if "T4" in mod][0])
#             select[self.num_fake_event_types+3] = True
#         if any("T5" in s for s in modalities):
#             iseicu = True
#             eicu_idx.append([index for index, mod in enumerate(modalities) if "T5" in mod][0])
#             select[self.num_fake_event_types+4] = True
#         if iseicu:
#             codes = codes.to(self.device)
#             types = types.to(self.device)
#             timestamps = timestamps.to(self.device)
#             ages = ages.to(self.device)
#             genders = genders.to(self.device)
#             ethnicities = ethnicities.to(self.device)

#             mask = (types != 0)
#             mask = torch.einsum("ab,ac->abc", mask, mask)
#             """ embedding """
#             # [# admissions, # batch_codes, embedding_size]
#             codes_emb = self.code_embedding(codes)
#             types_emb = self.type_embedding(types)
#             timestamps_emb = self.timestamp_embedding(timestamps) / math.sqrt(self.embedding_size)
#             ages_emb = self.age_embedding(ages)
#             genders_emb = self.gender_embedding(genders)
#             ethnicities_emb = self.ethnicity_embedding(ethnicities)
#             demographics_emb = (ages_emb + genders_emb + ethnicities_emb).unsqueeze(1)
#             emb = codes_emb + types_emb + timestamps_emb + demographics_emb
#             # sum every embedding
#             emb[mask.sum(dim=-1) == 0] = 0

#             """ transformer """
#             x = emb
#             for transformer in self.transformer:
#                 x = transformer(x, mask)  # [# admissions, # batch_codes, embedding_size]
#             x = self.dropout(x)
#             x = self.projector(x)  # [# admissions, # batch_codes, hidden_size]
#         import pdb; pdb.set_trace()

#         return embeddings
