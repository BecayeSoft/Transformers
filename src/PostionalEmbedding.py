import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):

    def __init__(self, embedding_dim, max_sequence_length) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length

    def forward(self):
        # Initalize the even and odd embeddings
        even_i = torch.arange(0, self.embedding_dim).float()
        # odd_i = torch.arange(0, self.embedding_dim)

        # Compute the denominator (same for odd and even)
        denominator = torch.pow(10000, (even_i-1/self.embedding_dim))

        # Initalize the position tensor
        position = torch.arange(0, self.max_sequence_length)
        # Reshape it in 2D  (e.g. [0, 1, ...]) -> [[0], [1], ...])
        position = position.reshape(-1, 1)

        # Compute the postion embeddings
        even_PE = torch.sin(position / denominator)
        odd_PE  = torch.cos(position / denominator)

        # Stack even and odd embeddings
        PE = torch.stack([even_PE, odd_PE], dim=-1)

        # Flatten the last dimension (See notebook)
        PE = torch.flatten(PE, start_dim=1, end_dim=-1)

        return PE
