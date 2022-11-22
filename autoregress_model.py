import torch

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
class smooth_histogram(nn.Module):
    # the distribution might be too large
    # only return the probability for x
    def __init__(self, n_values, *args, **kwargs):
        super(smooth_histogram, self).__init__()
        self.zeros = torch.nn.Parameter(torch.zeros(n_values), requires_grad=True)
    def forward(self, x):
        return F.softmax(self.zeros)[x]

    def display(self, n_values, train_loader):
        x = torch.linspace(0, n_values - 1, 100).type(torch.int64)
        y = self.forward(x)
        plt.hist(train_loader.dataset.xs.flatten(), bins=n_values)
        plt.plot(x, y.detach().numpy() * len(train_loader.dataset))
        # plt.ylim(0, 0.5)
        # print(train_loader.dataset.xs.flatten())
        plt.show()
class mixture_logistics(nn.Module):
    def __init__(self, n_values, n_logistics=4, **kwargs) -> None:
        super().__init__()
        self.pi = nn.Parameter(1 / n_logistics * torch.ones(n_logistics), requires_grad=True)
        self.mu = nn.Parameter(torch.linspace(0, n_values, n_logistics) * torch.ones(n_logistics), requires_grad=True)
        self.st = nn.Parameter(torch.ones(n_logistics), requires_grad=True)
    def forward(self, x):
        y = torch.zeros_like(x, dtype=torch.float32)
        for i in range(self.pi.shape[0]):
            y += F.softmax(self.pi)[i] * (torch.sigmoid((x + 0.5 - self.mu[i]) / self.st[i]) - torch.sigmoid((x - 0.5 - self.mu[i]) / self.st[i]))
        return y
        
    def display(self, n_values, train_loader):
        x = torch.linspace(0, n_values, 100)
        y = self.forward(x)
        plt.hist(train_loader.dataset.xs.flatten(), bins=n_values)
        plt.plot(x, y.detach().numpy() * len(train_loader.dataset))
        # plt.ylim(0, 0.5)
        # print(train_loader.dataset.xs.flatten())
        plt.show()

if __name__ == "__main__":
    model = smooth_histogram(10)
