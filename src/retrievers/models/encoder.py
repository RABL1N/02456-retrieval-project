import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from transformers import BertTokenizer, BertModel

import torch

class TransformerEncoder(nn.Module):
    
    # def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int):
        
    #     self.n_layers = n_layers
    #     self.n_heads = n_heads
    #     self.emb_dim = emb_dim
    #     self.hidden_dim = hidden_dim
    #     self.feed_forward_dim = feed_forward_dim

    #     self.pooling_type = pooling_type
    #     self.setup_model()

    def __init__(self, pooling_type: str):

        super().__init__()
        self.pooling_type = pooling_type
        self.setup_model()


    def setup_model(self):
        """
        Instantiate the model.
        """

        self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.model = BertModel.from_pretrained("google-bert/bert-base-uncased")

        # raise NotImplementedError()

    def pool(self, x):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        if self.cfg.model.pooling_type == 'cls':
            return x[:,0,:]
        
        elif self.cfg.model.pooling_type == 'max':
            return torch.max(x, dim=1)[0]
        
        elif self.cfg.model.pooling_type == 'mean':
            return torch.mean(x,dim=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Run a forward pass on the model
        """
        raise NotImplementedError()
