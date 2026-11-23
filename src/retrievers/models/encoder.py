import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from transformers import BertTokenizer, BertModel

import torch

class TransformerEncoder(nn.Module):

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

    def pool(self, x, attention_mask):
        """
        @param x: (B, L, D) last hidden states
        @param attention_mask: (B, L) 1 = real token, 0 = padding
        Returns: (B, D)
        """
        if self.pooling_type == 'cls':
            return x[:, 0, :]

        mask = attention_mask.unsqueeze(-1).to(x.dtype)           # (B, L, 1)

        if self.pooling_type == 'mean':
            return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        if self.pooling_type == 'max':
            return x.masked_fill(mask == 0, float('-inf')).max(dim=1).values

        raise ValueError(f'Unknown pooling_type: {self.pooling_type}')

    def forward(self, inputs):
        """
        @param inputs: tokenizer output (dict of tensors already on the model's device)
        Returns: (B, D) pooled embeddings
        """
        out = self.model(**inputs)
        return self.pool(out.last_hidden_state, inputs['attention_mask'])