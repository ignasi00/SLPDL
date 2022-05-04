# TODO: Functions to build and initialize the model

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, bias=True, dropout=None):
        super().__init__()

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = dropout

    def _attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -Inf)

        p_attn = F.softmax(scores, dim = -1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    # B = Batch size
    # W = Number of context words (left + right)
    # E = embedding_dim
    def forward(self, x, mask=None):
        # x shape is (B, W, E)
        q = self.q_proj(x)
        # q shape is (B, W, E)
        k = self.k_proj(x)
        # k shape is (B, W, E)
        v = self.v_proj(x)
        # k shape is (B, W, E)
        y, _ = self._attention(q, k, v, mask=mask, dropout=self.dropout)
        # y shape is (B, W, E)
        y = self.out_proj(y)
        # y shape is (B, W, E)
        return y

class TransformerLayer(nn.Module):
    def __init__(self, input_size, dim_feedforward=512, dropout_attn=0.1, dropout=0.1, num_heads_att=1):
        super().__init__()

        if num_heads_att == 1:
            self.self_attn = SelfAttention(input_size, dropout=dropout_attn)
        else:
            self.multi_head_attn = nn.ModuleList([SelfAttention(input_size, dropout=dropout_attn) for _ in range(num_heads_att)])
            self.multi_head_lin = nn.Linear(input_size * num_heads_att, input_size)

            self.self_attn = lambda u, mask=None : self.multi_head_lin( torch.cat( [f(u) for f in self.multi_head_attn], dim=-1 ) )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(input_size, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_size)
        
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        src2 = self.self_attn(src, mask)
        # Residual Connection with Dropout
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Dropout with no residual connections
        src = self.dropout(F.relu(self.linear1(src)))

        src2 = self.linear2(src)
        # Residual Connection with Dropout
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class Transformer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, context_words=6, num_seq_transformer=1, num_heads_att=1, dim_feedforward=512, dropout_attn=0.1, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lin = nn.Linear(embedding_dim, num_embeddings, bias=False)
        
        transformerLayer = lambda : TransformerLayer(embedding_dim, num_heads_att=num_heads_att, dim_feedforward=dim_feedforward, dropout_attn=dropout_attn, dropout=dropout)
        att = nn.ModuleList(*[transformerLayer() for _ in range(num_seq_transformer)])
        def apply_att(x, mask=None):
            for net in att:
                 x = net(x, mask)
            return x
        self.att = lambda x, mask=None : apply_att(x, mask)
        
        self.position_embedding = nn.Parameter(torch.Tensor(context_words, embedding_dim))

    # B = Batch size
    # W = Number of context words (left + right)
    # E = embedding_dim
    # V = num_embeddings (number of words)
    def forward(self, input_, mask=None):
        # input shape is (B, W)

        e = self.emb(input_)
        # e shape is (B, W, E)
        
        u = e + self.position_embedding
        # u shape is (B, W, E)
        
        v = self.att(u, mask)
        # v shape is (B, W, E)
        
        x = v.sum(dim=1)
        # x shape is (B, E)
        
        y = self.lin(x)
        # y shape is (B, V)

        return y
