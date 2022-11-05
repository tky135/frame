import torch

import torch.nn as nn
import torch.nn.functional as F

class smooth_histogram(nn.Module):
    # the distribution might be too large
    # only return the probability for x
    def __init__(self, n_values, *args, **kwargs):
        super(smooth_histogram, self).__init__()
        self.zeros = torch.nn.Parameter(torch.zeros(n_values), requires_grad=True)
    def forward(self, x):
        return F.softmax(self.zeros)[x]


if __name__ == "__main__":
    model = smooth_histogram(10)
