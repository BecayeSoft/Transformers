import torch
from torch import nn
import torch.nn.functional as F
import math



def scaled_dot_product(Q, K, V, mask=None):
    """Compute the attention weights and output values"""
    head_dim = Q.size()[-1]
    
    scaled = torch.matmul(
        Q, K.transpose(-2, -1)
    ) / math.sqrt(head_dim)

    if mask is not None:
        mask = torch.full(scaled.shape, -torch.inf)
        mask = torch.triu(mask, diagonal=1)
        scaled += mask

    weights = F.softmax(scaled, dim=-1)
    values = torch.matmul(weights, V)
    
    return values
    

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.linear_layer_in = nn.Linear(input_dim, output_dim*3)
        self.linear_layer_out = nn.Linear(input_dim, output_dim)


    def forward(self, X, mask=None) -> torch.Tensor:
        """Take in an input, extract Q, K, V, 
        compute the attention and return the output."""
        batch_size, seq_length, _ = X.size()

        # ------- Extract the Query, Key and Value vector ------- #
        
        QKV = self.linear_layer_in(X)
        
        # Reshape the tensor into 3 for Query, Key and Value vectors 
        QKV = QKV.reshape(
            batch_size, self.num_heads, seq_length, self.head_dim * 3
        )
        Q, K, V = QKV.chunk(3, dim=-1)

        # ------- Compute the attention ------- #
        
        values = scaled_dot_product(Q, K, V)
        
        # Concat the heads output
        values = values.reshape(
            batch_size, seq_length, self.head_dim*self.num_heads
        )

        # ------- Compute the output ------- #
        output = self.linear_layer_out(values)

        return output

