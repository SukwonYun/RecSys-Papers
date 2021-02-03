
import torch

from layer import FeaturesLinear, MultiLayerPerceptron, FeaturesEmbedding

class WideAndDeep(torch.nn.Module):
    def __init__ (self, field_dims, embed_dims, mlp_dims, dropout):
        super().__init__()                        
        self.linear = FeaturesLinear(field_dims)            
        self.embedding = FeaturesEmbedding(field_dims, embed_dims)
        self.embed_output_dim = len(field_dims) * embed_dims
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        
    def forward(self, x):
        """
        Parameters
        ----------
        x: Long tensor of size (batch_size, num_fields)
        """
        
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))




