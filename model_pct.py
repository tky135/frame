
import torch.nn as nn
import torch
from torch.nn import functional as F

class PCT(nn.Module):
    def __init__(self, in_channels, n_category) -> None:
        super().__init__()
        self.n_category = n_category
        
    def forward(self, x):
        # x: [B, N, C]
        return x