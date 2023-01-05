
import torch.nn as nn
import torch
from torch.nn import functional as F

class PCT(nn.Module):
    def __init__(self, n_category, in_channels=3, **kwargs) -> None:
        super().__init__()
        self.n_category = n_category
        self.linear1 = LBR(in_channels, 128)
        self.sa1 = self_attention(128, 128)
        self.sa2 = self_attention(128, 128)
        self.sa3 = self_attention(128, 128)
        self.sa4 = self_attention(128, 128)

        self.linear2 = LBR(128 * 4, 1024)
        self.cat_embed = nn.Linear(16, 128)

        self.linear3 = LBR(3 * 1024 + 128, 256)
        self.dropout = nn.Dropout(0.5)
        self.linear4 = LBR(256, 256)
        self.linear5 = nn.Linear(256, n_category)
    def forward(self, x, cat):
        # x: [B, N, C]
        B, N, C = x.shape
        # input embedding
        x = self.linear1(x)

        # self attention
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        # cat
        x = torch.cat([x1, x2, x3, x4], dim=-1)
        x = self.linear2(x)

        # mean-max pooling & embedding
        x_global = torch.cat([torch.mean(x, dim=1), torch.max(x, dim=1)[0], self.cat_embed(cat.float())], dim=-1)

        # repeat
        x_global = x_global.unsqueeze(1).repeat(1, N, 1)
        x = torch.cat([x, x_global], dim=-1)

        # output
        x = self.dropout(self.linear3(x))
        x = self.linear4(x)
        x = self.linear5(x)
        return x
class LBR(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
    def forward(self, x):
        # x: [B, N, C]
        x = F.relu(self.bn1(self.linear1(x).transpose(1, 2))).transpose(1, 2)
        return x
class self_attention(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.k_linear = nn.Linear(in_channels, in_channels // 4)
        self.q_linear = nn.Linear(in_channels, in_channels // 4)
        self.v_linear = nn.Linear(in_channels, out_channels)
    def forward(self, x):
        # x: [B, N, C]
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        # [B, N, N]
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / (k.shape[-1] ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [B, N, C]
        out = torch.matmul(attention, v)
        return out

class sample_and_group(nn.Module):
    def __init__(self) -> None:
        super().__init__()