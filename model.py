from turtle import forward
import torch.nn as nn
import torch
from torch.nn import functional as F
import torchvision.models as models
from util import sample_and_group 


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
    def __init__(self, input_channels=3, n_category=10, pretrained=True, **kwargs):
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
    def __init__(self, input_channels=3, n_category=10, pretrained=True, **kwargs):
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

class FCN(nn.Module):
    def __init__(self, n_category, pretrained=False, *args, **kwargs):
        super().__init__()
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=pretrained)
        if n_category != 21:
            self.net.classifier[4] = nn.Conv2d(in_channels=self.net.classifier[4].in_channels, out_channels=n_category, kernel_size=self.net.classifier[4].kernel_size, stride=self.net.classifier[4].stride)
    def forward(self, x):
        return self.net(x)['out']

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
# class 


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x
class TestPCCls(nn.Module):
    def __init__(self, n_category, **kwargs):
        super().__init__()
        self.linear1 = nn.Linear(3, 128)
        self.linear2 = nn.Linear(128, n_category)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = torch.max(x, 1)[0]
        x = self.linear2(x)
        return x
class TestPCSeg(nn.Module):
    def __init__(self, n_category, **kwargs) -> None:
        super().__init__()
        self.linear1 = nn.Linear(3, 128)
        
class Pct(nn.Module):
    def __init__(self, n_category=40, **kwargs):
        super(Pct, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last()

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, n_category)

    def forward(self, x):
        # xyz = x.permute(0, 2, 1)
        xyz = x
        batch_size, _, _ = x.size()
        # B, D, N
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
if __name__ == "__main__":
    # import data
    # vgg = VGG()
    # google = GoogLeNet()
    # alex = AlexNet()
    # res = ResNet18()
    # mobile = MobileNetV2()
    # print(get_n_params(mobile))
    alex = MobileNetV2()
    print(get_n_params(alex))
    # resnet = models.resnet18(pretrained=True)
    # print(resnet)
    # raise Exception("break")
    # alexnet = models.alexnet(pretrained=True)
    # x = torch.zeros((1, 3, 224, 224))
    # x = data.train_augs(x).unsqueeze(0)
    # y = alexnet(x)
    # print(y.shape)
    # alexnet = AlexNet(n_category=4)

