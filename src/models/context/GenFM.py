# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import numpy as np

from models.BaseContextModel import ContextCTRModel
from models.context.FM import FMBase

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


class GenFMBase(FMBase):    
    def _define_init(self, args, corpus):
        super()._define_init(args, corpus)
        self.num_fields = len(self.context_features)
        self.gen_encoder = GenEncoder(self.num_fields, self.vec_size) # vec_size is args.emb_size

    def forward(self, feed_dict):

        # fm_vectors: [Batch, Item_Num, Num_Fields, Emb_Size]
        # linear_value: [Batch, Item_Num]
        fm_vectors, linear_value = self._get_embeddings_FM(feed_dict)
        
        # gen_vectors: [Batch, Item_Num, Num_Fields, Emb_Size]
        gen_vectors = self.gen_encoder(fm_vectors)
        
        num_fields = fm_vectors.shape[2]
        
        row, col = torch.triu_indices(num_fields, num_fields, offset=1, device=fm_vectors.device)
        
        interaction_vec = gen_vectors[:, :, row, :] * fm_vectors[:, :, col, :]
        
        # Sum over embedding dim and pairs -> [Batch]
        fm_prediction = interaction_vec.sum(dim=(-2,-1))
        
        predictions = linear_value + fm_prediction
        
        return {'prediction': predictions}

class GenFMCTR(ContextCTRModel, GenFMBase):
    reader, runner = 'ContextReader', 'CTRRunner'
    extra_log_args = ['emb_size', 'loss_n']

    @staticmethod
    def parse_model_args(parser):
        parser = FMBase.parse_model_args_FM(parser)
        return ContextCTRModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        ContextCTRModel.__init__(self, args, corpus)
        self._define_init(args, corpus)

    def forward(self, feed_dict):
        out_dict = GenFMBase.forward(self, feed_dict)
        out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
        out_dict['label'] = feed_dict['label'].view(-1)
        return out_dict