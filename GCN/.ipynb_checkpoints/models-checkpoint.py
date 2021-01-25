#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, n_feat, n_hidden, n_class, dropout):
        super(GCN, self).__init__()
        
        self.gc1 = GraphConvolution(n_feat, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_class)
        self.dropout = dropout
        
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x,adj)
        return F.log_softmax(x, dim=1)

