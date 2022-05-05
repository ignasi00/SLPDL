# TODO: Functions to build and initialize model with different options for the lab

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_context_words=6, vector=None, shared_embedding=False):
        # By default, num_context_words=6 in order to be fully compatible with the original code from José Adrién Rodríguez Fonollosa
        super().__init__()
        
        if shared_embedding:
            self.embedding = nn.Parameter(torch.rand(num_embeddings, embedding_dim), requires_grad=True)
            self.emb = lambda x : F.embedding(x, self.embedding, padding_idx=0)
            self.lin = lambda x : F.linear(x, self.embedding)
        else:
            self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
            self.emb.weight.data[0] = self.emb.weight.data[1:].sum(dim=0) / num_embeddings
            self.lin = nn.Linear(embedding_dim, num_embeddings, bias=False)
        
        
        if vector is True:
            weights = torch.rand(num_context_words, embedding_dim).unsqueeze(dim=0)
        else:
            weights = torch.ones(num_context_words).unsqueeze(dim=0).unsqueeze(dim=2)

        self.weights = nn.Parameter(weights, requires_grad=True)
        # To manually assign weigths and freeze them, the instance should be modified (not the class)
        # Initiallizations should be done on the script, the constructor only define the shape

    # B = Batch size
    # W = Number of context words (left + right)
    # E = embedding_dim
    # V = num_embeddings (number of words)
    def forward(self, input):
        # input shape is (B, W)

        e = self.emb(input)
        # e shape is (B, W, E)

        # dot product through dim=1 to apply weights:
        u = (e * self.weights).sum(dim=1)
        # u shape is (B, E)

        z = self.lin(u)
        # z shape is (B, V)

        return z
