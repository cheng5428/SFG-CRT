# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import numpy as np

from models.context.DeepFM import DeepFMBase, DeepFMCTR
from models.context.WideDeep import WideDeepCTR

class GenEncoder(nn.Module):
    """
    SFG Encoder: 
    Transforms Raw Embeddings [B, I, F, D] -> Generated Embeddings [B, I, F, D]
    Utilizing global context via a wide linear layer + ReLU.
    """
    def __init__(self, num_fields, emb_size):
        super(GenEncoder, self).__init__()
        self.num_fields = num_fields
        self.emb_size = emb_size
        self.input_dim = num_fields * emb_size
        
        self.projection = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.activation = nn.ReLU() 
        
        # Learnable scaling factor
        self.gamma = nn.Parameter(torch.ones(1))
        
        # 初始化
        nn.init.xavier_normal_(self.projection.weight)

    def forward(self, raw_emb):
        # raw_emb shape: [B, I, F, D]
        original_shape = raw_emb.shape

        # Flatten: [B, I, F, D] -> [B*I, F*D]
        flat_input = raw_emb.view(-1, self.input_dim)
        
        # Encode
        out = self.projection(flat_input)
        out = self.activation(out)
        
        # Scale and Reshape back to [B, I, F, D]
        gen_emb = (out * self.gamma).view(original_shape)
        return gen_emb

class GenDeepFMBase(DeepFMBase):
    def __init__(self, args, corpus):
        # 获取特征域数量
        self.num_fields = len(corpus.feature_max)
        self.gen_encoder = GenEncoder(self.num_fields, args.emb_size)

    def forward(self, feed_dict):
        # context_vectors: [Batch, Item_Num, Num_Fields, Emb_Size] 
        # linear_vectors: [Batch, Item_Num] 
        context_vectors, linear_vectors = self._get_embeddings_FM(feed_dict)

        # 输入: [B, I, F, D] -> 输出: [B, I, F, D]
        gen_vectors = self.gen_encoder(context_vectors)

        num_fields = context_vectors.shape[2]
        
        row, col = torch.triu_indices(num_fields, num_fields, offset=1, device=context_vectors.device)
        
        # gen_vectors[:, :, row, :] 形状为 [B, I, Num_Pairs, D]
        interaction_vec = gen_vectors[:, :, row, :] * context_vectors[:, :, col, :]
        
        # [B, I, Num_Pairs, D] -> [B, I]
        fm_prediction = interaction_vec.sum(dim=(-2,-1))
        
        fm_prediction = fm_prediction + linear_vectors
        
        # Deep
        deep_prediction = self.deep_layers(context_vectors.flatten(start_dim=-2)).squeeze(-1)
        
        predictions = fm_prediction + deep_prediction
        
        return {'prediction': predictions}


class GenDeepFMCTR(WideDeepCTR, GenDeepFMBase):
    reader, runner = 'ContextReader', 'CTRRunner'
    extra_log_args = ['emb_size', 'layers', 'loss_n']
    
    def __init__(self, args, corpus):
        WideDeepCTR.__init__(self, args, corpus)
        GenDeepFMBase.__init__(self, args, corpus)

    def forward(self, feed_dict):
        out_dict = GenDeepFMBase.forward(self, feed_dict)
        out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
        out_dict['label'] = feed_dict['label'].view(-1)
        return out_dict