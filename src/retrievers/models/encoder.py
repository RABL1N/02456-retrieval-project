import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.utils.checkpoint import checkpoint

from transformers import BertTokenizer, BertModel


# Our own Transformer encoder, with exactly the BERT-base architecture so the pretrained
# BERT weights can be loaded into it (see load_hf_bert_weights).

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        self.v = nn.Linear(emb_dim, emb_dim)
        self.o = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        """
        @param x: (B, L, D)
        @param attention_mask: (B, L) 1 = real token, 0 = padding
        Returns: output (B, L, D), attention weights (B, H, L, L)
        """
        # Shapes use B = batch size, L = number of tokens, D = 768 (embedding size), H = 12 heads, d = 64 = D/H (per-head size).
        B, L, _ = x.shape
        split = lambda t: t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)   # (B, H, L, d)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))

        scores = q @ k.transpose(-2, -1) / sqrt(self.head_dim)                        # (B, H, L, L)
        scores.masked_fill_(attention_mask[:, None, None, :] == 0, float('-inf'))  # no attending to padding (in place: saves a copy)
        attn = scores.softmax(dim=-1)

        out = (self.dropout(attn) @ v).transpose(1, 2).reshape(B, L, -1)            # concatenate heads
        return self.o(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, feed_forward_dim: int, dropout: float = 0.1, eps: float = 1e-12):
        super().__init__()
        self.attention = MultiHeadAttention(emb_dim, n_heads, dropout)
        self.ln1 = nn.LayerNorm(emb_dim, eps=eps)
        self.ff1 = nn.Linear(emb_dim, feed_forward_dim)
        self.ff2 = nn.Linear(feed_forward_dim, emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        a, attn = self.attention(x, attention_mask)
        x = self.ln1(x + self.dropout(a))                                   # residual + LayerNorm (post-LN, as BERT)
        x = self.ln2(x + self.dropout(self.ff2(F.gelu(self.ff1(x)))))       # feed-forward + residual + LayerNorm
        return x, attn


class BertEncoder(nn.Module):
    def __init__(self, vocab_size: int = 30522, max_len: int = 512, n_types: int = 2,
                 emb_dim: int = 768, n_heads: int = 12, n_layers: int = 12, feed_forward_dim: int = 3072,
                 dropout: float = 0.1, eps: float = 1e-12):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_len, emb_dim)          # learned positional representations
        self.type_emb = nn.Embedding(n_types, emb_dim)         # segment A/B
        self.emb_ln = nn.LayerNorm(emb_dim, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            EncoderLayer(emb_dim, n_heads, feed_forward_dim, dropout, eps) for _ in range(n_layers)
        )
        self.grad_ckpt = False

    def gradient_checkpointing_enable(self):
        """Recompute layer activations in the backward pass instead of storing them (saves memory)."""
        self.grad_ckpt = True

    def forward(self, input_ids, attention_mask, token_type_ids=None, return_attn=False):
        """
        Returns: last hidden states (B, L, D), plus a list of per-layer attention weights if return_attn.
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device)[None, :]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = self.token_emb(input_ids) + self.pos_emb(positions) + self.type_emb(token_type_ids)
        x = self.dropout(self.emb_ln(x))

        attns = []
        for layer in self.layers:
            if self.grad_ckpt and self.training:
                x, a = checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x, a = layer(x, attention_mask)
            if return_attn:                     # only keep the (B, H, L, L) maps when asked: ~0.8 GB per layer at 64 x 512
                attns.append(a)
            del a
        return (x, attns) if return_attn else x


def load_hf_bert_weights(model: BertEncoder, hf_model: BertModel) -> BertEncoder:
    """Copy the weights of pretrained Hugging Face BertModel into our BertEncoder (same architecture)."""
    sd = hf_model.state_dict()
    names = {'token_emb': 'embeddings.word_embeddings',
             'pos_emb':   'embeddings.position_embeddings',
             'type_emb':  'embeddings.token_type_embeddings',
             'emb_ln':    'embeddings.LayerNorm'}
    for i in range(len(model.layers)):
        ours, theirs = f'layers.{i}.', f'encoder.layer.{i}.'
        names.update({ours + 'attention.q': theirs + 'attention.self.query',
                      ours + 'attention.k': theirs + 'attention.self.key',
                      ours + 'attention.v': theirs + 'attention.self.value',
                      ours + 'attention.o': theirs + 'attention.output.dense',
                      ours + 'ln1':         theirs + 'attention.output.LayerNorm',
                      ours + 'ff1':         theirs + 'intermediate.dense',
                      ours + 'ff2':         theirs + 'output.dense',
                      ours + 'ln2':         theirs + 'output.LayerNorm'})
    new_sd = {f'{o}.{p}': sd[f'{t}.{p}'] for o, t in names.items() for p in ('weight', 'bias') if f'{t}.{p}' in sd}
    model.load_state_dict(new_sd, strict=True)          # fails loudly if any weight is missing or unexpected
    return model


# Retrieval encoder: tokenizer + Transformer + pooling

class TransformerEncoder(nn.Module):

    def __init__(self, pooling_type: str, implementation: str = 'own'):
        """
        @param implementation: 'own' = our BertEncoder with pretrained BERT weights loaded in,
                               'hf'  = Hugging Face BertModel (needed to load checkpoints trained with it)
        """
        super().__init__()
        self.pooling_type = pooling_type
        self.implementation = implementation
        self.setup_model()


    def setup_model(self):
        """
        Instantiate the model.
        """
        name = "google-bert/bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(name)
        hf = BertModel.from_pretrained(name, add_pooling_layer=False)

        if self.implementation == 'hf':
            self.model = hf
        elif self.implementation == 'own':
            c = hf.config
            self.model = BertEncoder(vocab_size=c.vocab_size, max_len=c.max_position_embeddings,
                                     n_types=c.type_vocab_size, emb_dim=c.hidden_size,
                                     n_heads=c.num_attention_heads, n_layers=c.num_hidden_layers,
                                     feed_forward_dim=c.intermediate_size,
                                     dropout=c.hidden_dropout_prob, eps=c.layer_norm_eps)
            load_hf_bert_weights(self.model, hf)
        else:
            raise ValueError(f'Unknown implementation: {self.implementation}')

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

    def hidden_states(self, inputs):
        """(B, L, D) last hidden states for tokenizer output `inputs`."""
        if self.implementation == 'hf':
            return self.model(**inputs).last_hidden_state
        return self.model(inputs['input_ids'], inputs['attention_mask'], inputs.get('token_type_ids'))

    def forward(self, inputs):
        """
        @param inputs: tokenizer output (dict of tensors already on the model's device)
        Returns: (B, D) pooled embeddings
        """
        return self.pool(self.hidden_states(inputs), inputs['attention_mask'])
