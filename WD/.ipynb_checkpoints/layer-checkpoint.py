#!/usr/bin/env python
# coding: utf-8


import numpy as np
import torch
import torch.nn.functional as F

class FeaturesLinear(torch.nn.Module):
    
    def __init__(self, field_dims, output_dim = 1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype = np.long)
        
    def forward(self, x):
        """
        Parameter
        ---------
        x : Long tensor of size (batch_size, num_fields)
        
        """
        #new_tensor enables locating x in same gpu
        x = x + x.new_tensor(self.offsets).unsqueeze(0)        
        
        return torch.sum(self.fc(x), dim = 1) + self.bias #summation in row-wise
    
    
class FeaturesEmbedding(torch.nn.Module):
    
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        
        #Starting point of each field
        self.offsets = np.array((0,*np.cumsum(field_dims)[:-1]), dtype =np.long) 
        
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        
    def forward(self, x):
        """
        Parameter
        ---------
        x : Long tensor of size (btach_size, num_fields)
        
        """
        #new_tensor enables locating x in same gpu
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        
        return self.embedding(x)
    

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Parametes
        ---------
        x : Float tensor of size (batch size, num_fields, embed_dim)
        """
        return self.mlp(x)