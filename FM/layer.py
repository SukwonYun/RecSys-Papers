# coding: utf-8


import numpy as np
import torch
import torch.nn.functional as F

class FactorizationMachine(torch.nn.Module):
    
    def __init__(self, reduce_sum = True):
        super().__init__()
        self.reduce_sum = reduce_sum
        
    def forward(self, x):
        """
        Parameters
        ----------
        x : Float tensor of size (batch_size, num_fields, embed_dim)
        """
        
        square_of_sum = torch.sum(x, dim = 1) ** 2 #summation in column direction
        sum_of_square = torch.sum(x ** 2, dim = 1)
        ix = square_of_sum - sum_of_square
        
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
            
        return 0.5 * ix
    
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
        
        return torch.sum(self.fc(x), dim = 1) + self.bias

