import torch
import torch.nn as nn

class LayerNormalization:

    def __init__(self, sequence_length, batch_size, embedding_dim) -> None:
        self.sequence_length = sequence_length
        self.batch_size = batch_size 
        self.embedding_dim = embedding_dim

    def forward(self, X):
        m = X.size()[-1]
        epsilon = 1e-5

        # Init gamma and beta
        params_shape = X.size()[-2:] 
        gamma = nn.Parameter(torch.ones(params_shape))
        beta = nn.Parameter(torch.zeros(params_shape))

        # Dynamically get last two dimensions of tensor for computation
        dims = [-(i+1) for i in range(len(params_shape))]

        # Mean, variance, normalization, scale and shift
        mu = (1/m) * torch.sum(X, dim=dims, keepdim=True)
        sigma2 = (1/m) * torch.sum((X - mu) ** 2, dim=dims, keepdim=True)
        X_norm = (X - mu) / torch.sqrt(sigma2 + epsilon)
        Y = gamma * X_norm + beta

        return Y

