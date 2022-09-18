import struct
from cv2 import split
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import metrics

import torch
import torch.nn.functional as F

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def readMNIST(file):
    f = open(file, "rb")
    magic, num = struct.unpack(">ii", f.read(8))

    if magic == 2049:
        # lable file
        y = np.zeros(num, dtype=np.int32)
        for i in range(num):
            y[i] = struct.unpack("b", f.read(1))[0]
        return y
    elif magic == 2051:
        row, col = struct.unpack(">ii", f.read(8))
        x = np.zeros((num, row, col), dtype=np.int32)
        for k in range(num):
            for j in range(col):
                for i in range(row):
                    x[k, j, i] = struct.unpack("B", f.read(1))[0]
        return x
    else:
        return None
    
    f.close()

def clean_cifar():
    path = os.path.join("/data", "cifar-10")
    f = open(os.path.join(path, "trainLabels.csv"))
    g = open(os.path.join(path, "train.csv"), 'w')
    f.readline()
    g.write("image,label\n")
    for line in f.readlines():
        idd, label = line.split(',')
        g.write("train/" + idd + ".png," + label)

    f.close()
    g.close()
    h = open(os.path.join(path, "test.csv"), 'w')
    h.write("image\n")
    for img in os.listdir(os.path.join(path, "test")):
        h.write("test/" + img + '\n')
    h.close()
        
def plot_confusion_matrix(confusion_matrix, labels):
    display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = labels)
    display.plot()
    plt.savefig("confusion_matrix.jpg")
if __name__ == "__main__":
    plot_confusion_matrix(np.array([[0, 1], [2, 3]]), ["yes", "no"])


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    print("knn_point")
    print(new_xyz.shape)
    print(xyz.shape)
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    B, N, D = point.shape
    new_point_idx = []
    for xyz in point:
        centroids = torch.zeros((npoint,), device=torch.device("cuda"))
        distance = torch.ones((N,), device=torch.device("cuda")) * 1e10
        farthest = torch.randint(0, N, (1,), device=torch.device("cuda"))
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.argmax(distance, -1)
        new_point_idx.append(centroids.type(torch.long))
    return torch.stack(new_point_idx, dim=0)

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint 
    xyz = xyz.contiguous()
    print("before fps", xyz.shape)
    fps_idx = farthest_point_sample(xyz, npoint).long() # [B, npoint]
    print(fps_idx.shape)
    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points