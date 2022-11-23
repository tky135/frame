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
    
class MADE2d(nn.Module):
    def __init__(self, n_values, **kwargs) -> None:
        super().__init__()
        self.e1 = nn.Parameter(torch.zeros(n_values), requires_grad=True)
        self.e21 = nn.Linear(n_values, 128, bias=True)
        self.e22 = nn.Linear(128, 128, bias=True)
        self.e23 = nn.Linear(128, 128, bias=True)
        self.e24 = nn.Linear(128, n_values, bias=True)
        self.e2 = nn.Sequential(self.e21, nn.ReLU(), self.e22, nn.ReLU(), self.e23, nn.ReLU(), self.e24)
        self.n_values = n_values
    def forward(self, x):
        one_hot_x = F.one_hot(x, self.n_values).type(torch.float32)
        y1 = F.log_softmax(self.e1, dim=0)[x[:, 0]]
        # print(self.e1.unsqueeze(0).repeat(x.shape[0], 1))
        y2 = F.log_softmax(self.e2(one_hot_x[:, 0, :]), dim=1)[torch.arange(x.shape[0]), x[:, 1]]
        # y2 = F.log_softmax(self.e2(torch.cat([one_hot_x[:, 0, :], self.e1.unsqueeze(0).repeat(x.shape[0], 1)], dim=1)), dim=1)[torch.arange(x.shape[0]), x[:, 1]]
        return y1 + y2

    def display(self, *args, **kwargs):
        xs = []
        ys = []
        weights = []
        for i in range(self.n_values):
            for j in range(self.n_values):
                xs.append(i)
                ys.append(j)
                weights.append(self.forward(torch.tensor([[i, j]]).type(torch.int64)).item())
        plt.hist2d(xs, ys, bins=self.n_values, weights=weights)
        plt.show()
class _MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True) -> None:
        """
        mask: shape (out_features, in_features)
        """
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))
        self.mask.data = mask
    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

class MADE(nn.Module):
    def __init__(self, n_values, n_dims, hidden = [784, 784], **kwargs) -> None:
        """
        n_values: number of values for each dimension (e.g. 256 for MNIST images)
        n_dims: number of dimensions (e.g. 784 for MNIST images)
        hidden: list of hidden layer sizes
        """
        super().__init__()
        self.n_values = n_values
        self.n_dims = n_dims
        self.layers = [n_values * n_dims] + hidden + [n_values * n_dims]
        
        # check hidden layer sizes
        for i in hidden:
            if i % n_dims != 0:
                raise ValueError("Hidden layer size must be a multiple of n_dims")

        # label each node with the dimension it is conditioned on
        self.labels = []

        # calculate input layer labels
        self.labels.append(torch.arange(n_dims).unsqueeze(-1).repeat(1, n_values).flatten())
        # randomly allocate labels for each hidden layer
        for l in hidden:
            self.labels.append(torch.arange(n_dims).repeat(l // n_dims))
        # calculate output layer labels
        self.labels.append(torch.arange(n_dims).unsqueeze(-1).repeat(1, n_values).flatten())

        # determine masks
        masks = []

        # type 1 masks
        for i in range(len(self.layers) - 2):
            masks.append(self.labels[i + 1].unsqueeze(1) >= self.labels[i].unsqueeze(0))
        # type 2 mask
        masks.append(self.labels[-1].unsqueeze(1) > self.labels[-2].unsqueeze(0))

        # create network
        net = []
        for i in range(len(self.layers) - 1):
            net.append(_MaskedLinear(self.layers[i], self.layers[i + 1], masks[i]))
            if i < len(self.layers) - 2:
                net.append(nn.ReLU())
        self.net = nn.Sequential(*net)
        # import numpy as np
        # for i in range(len(masks)):
        #     np.savetxt("mask.tky" + str(i), masks[i].type(torch.uint8).numpy().astype(np.uint8), fmt="%d")
        # for i in range(len(self.labels)):
        #     np.savetxt("labels.tky" + str(i), self.labels[i].numpy().astype(np.uint8), fmt="%d")
        # raise Exception("break")
    def forward(self, x):
        """
        x: shape (batch_size, n_dims)
        Returns the log probability of x
        """
        # one-hot encode x
        one_hot_x = F.one_hot(x, self.n_values).type(torch.float32)
        # flatten one-hot encoded x
        flat_one_hot_x = one_hot_x.view(-1, self.n_dims * self.n_values)
        # pass through network
        y = self.net(flat_one_hot_x)
        # reshape output
        y = y.view(-1, self.n_dims, self.n_values)
        # calculate log probabilities
        y = F.log_softmax(y, dim=2)[torch.arange(x.shape[0]).unsqueeze(-1).repeat(1, x.shape[1]), torch.arange(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1), x]
        return y.mean(dim=1)
    def display(self, *args, **kwargs):
        xs = []
        ys = []
        weights = []
        for i in range(self.n_values):
            for j in range(self.n_values):
                xs.append(i)
                ys.append(j)
                weights.append(self.forward(torch.tensor([[i, j]]).type(torch.int64)).item())
        plt.hist2d(xs, ys, bins=self.n_values, weights=weights)
        plt.show()
    def sample(self):
        for j in range(16):
            samples = torch.zeros(self.n_dims).type(torch.int64)
            for i in range(self.n_dims):
                # get conditional probabilities
                x = samples.unsqueeze(0)
                # sample from conditional probabilities
                one_hot_x = F.one_hot(x, self.n_values).type(torch.float32)
                # flatten one-hot encoded x
                flat_one_hot_x = one_hot_x.view(-1, self.n_dims * self.n_values)
                # pass through network
                y = self.net(flat_one_hot_x)
                # reshape output
                y = y.view(-1, self.n_dims, self.n_values)
                # calculate log probabilities
                y = F.log_softmax(y, dim=2)

                probs = torch.exp(y)
                # print(probs)
                samples[i] = torch.multinomial(probs[0, i], 1)
            samples = samples.reshape(28, 28)
            ax = plt.subplot(4, 4, j + 1)
            ax.imshow(samples.detach().numpy())
            ax.axis('off')
        plt.show()
        plt.savefig("samples.png")
if __name__ == "__main__":
    model = MADE(5, 8, hidden=[128, 128])
    model(torch.randint(0, 5, (10, 8)))
    print(model)