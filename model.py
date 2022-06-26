from turtle import forward
import torch.nn as nn
import torch
from torch.nn import functional as F
import torchvision.models as models
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.1)
    
class cnnMNIST(nn.Module):
    def __init__(self) -> None:
        super(cnnMNIST, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 4, 3), 
            nn.Conv2d(4, 8, 3), 
            nn.Flatten(),
            nn.Linear(4608, 128),
            nn.ReLU(), 
            nn.Dropout(),
            nn.Linear(128, 10), 
            nn.Softmax(dim=1)
        )
        self.model.apply(init_weights)
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.model(x)


class simpleLinear(nn.Module):
    def __init__(self) -> None:
        super(simpleLinear, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(470, 512),
            nn.ReLU(),
            nn.Linear(512, 64), 
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.model = self.model.apply(init_weights)
    def forward(self, x):
        return self.model(x)


class ResNet18(nn.Module):
    "11,181,642 parameters"
    def __init__(self, input_channels=3, n_category=10, pretrained=True):
        # input_dimension doesn't matter
        super().__init__()
        self.net = models.resnet18(pretrained=pretrained)
        self.net.fc = nn.Linear(self.net.fc.in_features, n_category, bias=True)
        nn.init.xavier_uniform_(self.net.fc.weight)
    def forward(self, x):
        y = self.net(x)
        return y


class AlexNet(nn.Module):
    "57,044,810 parameter"
    def __init__(self, input_channels=3, n_category=10, pretrained=True):
        super().__init__()
        self.net = models.alexnet(pretrained=pretrained)
        self.net.classifier[6] = nn.Linear(self.net.classifier[6].in_features, n_category, bias=True)
        nn.init.xavier_uniform_(self.net.classifier[6].weight)
        print(self.net)
    def forward(self, x):
        y = self.net(x)
        return y

class VGG(nn.Module):
    "128,807,306 parameters"
    def __init__(self, input_channels=3, n_category=10, pretrained=True):
        super().__init__()
        self.net = models.vgg11(pretrained=pretrained)
        self.net.classifier[6] = nn.Linear(self.net.classifier[6].in_features, n_category, bias=True)
        nn.init.xavier_uniform_(self.net.classifier[6].weight)
    def forward(self, x):
        y = self.net(x)
        return y
class GoogLeNet(nn.Module):
    "5,610,154 parameters"
    def __init__(self, input_channels=3, n_category=10, pretrained=True):
        super().__init__()
        self.net = models.googlenet(pretrained=pretrained)
        print(self.net)
        self.net.fc = nn.Linear(self.net.fc.in_features, n_category, bias=True)
        nn.init.xavier_uniform_(self.net.fc.weight)
    def forward(self, x):
        y = self.net(x)
        return y

class MobileNetV2(nn.Module):
    "2,236,682 parameters"
    def __init__(self, input_channels=3, n_category=10, pretrained=True):
        super().__init__()
        self.net = models.mobilenet_v2(pretrained=pretrained)
        self.net.classifier[1] = nn.Linear(self.net.classifier[1].in_features, n_category, bias=True)
        nn.init.xavier_uniform_(self.net.classifier[1].weight)
    def forward(self, x):
        y = self.net(x)
        return y
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
# class 
if __name__ == "__main__":
    # import data
    # vgg = VGG()
    # google = GoogLeNet()
    # alex = AlexNet()
    # res = ResNet18()
    # mobile = MobileNetV2()
    # print(get_n_params(mobile))
    alex = AlexNet()
    # resnet = models.resnet18(pretrained=True)
    # print(resnet)
    # raise Exception("break")
    # alexnet = models.alexnet(pretrained=True)
    # x = torch.zeros((1, 3, 224, 224))
    # x = data.train_augs(x).unsqueeze(0)
    # y = alexnet(x)
    # print(y.shape)
    # alexnet = AlexNet(n_category=4)

