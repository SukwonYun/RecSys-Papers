
import torch

from layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear

class FM(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum = True)
    
    def forward(self, x):
        """
        parameter
        ---------
        x : Long tensor of size (batch_size, num_fields)
        
        """
        
        x = self.linear(x) + self.fm(self.embedding(x))
        
        return torch.sigmoid(x.squeeze(1)) #exclude 1 in shape (ax1) -> (a)

