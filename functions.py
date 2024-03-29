import torch
import numpy as np

# This file contains loss functions and metrics for training and evaluation
# each function takes in y_pred and y, whatever their shapes are. 
def CEloss(y_pred, y):
    # celoss = torch.nn.CrossEntropyLoss() # cross entropy includes softmax
    # y_pred = torch.softmax(y_orig, dim=1)
    # print(y_pred.shape)
    # print(y.squeeze().shape)
    # print(set(y.detach().cpu().numpy().flatten()))
    # raise Exception("break")
    # y[y == 255] = 21 ### Tesing

    ### MUST MAKE SURE THE SHAPES ARE CORRECT
    y_pred = y_pred.permute(0, 2, 1)
    z = torch.nn.functional.cross_entropy(y_pred, y)
    if z == torch.inf:
        print("y_pred: ")
        print(y_pred)
        print("y_gold: ")
        # print(y_pred)
        print(y)
        # print(z_mid)
        raise Exception("infinity error")

    return z

def RMSElog(y_pred, y):
    y_pred = torch.clamp(y_pred, 1e-10, float('inf'))
    mse = torch.nn.MSELoss()
    return torch.sqrt(mse(torch.log(y_pred.reshape(-1)), torch.log(y)))

def class_acc(y_pred: torch.Tensor, y: torch.Tensor):
    """
    Average accuracy function for classification and semantic segmentation
    Input:
    y_pred: Tensor, prediction with shape [B, C, D1, D2, D3, ...]
    y     : Tensor, ground truth with shape [B, D1, D2, D3]

    """
    y_pred = y_pred.permute(0, 2, 1)
    z = torch.sum((torch.max(y_pred, dim=1)[1] == y)) / y.flatten().shape[0]

    return z

# def seg_acc(y_pred, y):
#     z = torch.sum((torch.max(y_pred, dim=1)[1] == y)) / (y_pred.shape[0] * y_pred.shape[1] * y_pred.shape[2])
# def acc_fn(y_pred, y):
#     mse = torch.nn.MSELoss()
#     return -torch.sqrt(mse(torch.log(y_pred.reshape(-1)), torch.log(y)))

def neg_log_likelihood(y_pred):
    return -torch.mean(y_pred)

# TODO 
def calculate_shape_IoU_np(y_pred, y):
    # cpu implementation
    # y_pred [B, C, H, W]
    # y      [B, H, W]
    y_pred, y = y_pred.detach().cpu().numpy(), y.cpu().numpy()
    y_pred = np.argmax(y_pred, axis=1)

    batch_ious = []
    for b in range(y.shape[0]):
        # average over each batch
        class_set = np.unique(y[b])        
        part_ious = []
        for part in class_set:
            I = np.sum(np.logical_and(y_pred[b] == part, y[b] == part))
            U = np.sum(np.logical_or(y_pred[b] == part, y[b] == part))
            iou = I / float(U)
            part_ious.append(iou)
        batch_ious.append(np.mean(part_ious)) # part IoU averaged
    return np.mean(batch_ious)


def instance_average_IoU(y_pred, y):
    # y_pred [B, C, D1, D2, ..., Dn]
    # y [B, D1, D2, ..., Dn]
    y_pred = y_pred.permute(0, 2, 1)
    y_pred = torch.argmax(y_pred, dim=1)
    B = y_pred.shape[0]
    batch_ious = []
    for b in range(B):
        part_set = torch.unique(y[b])
        part_ious = []
        for part in part_set:
            I = torch.sum(torch.logical_and(y_pred[b] == part, y[b] == part))
            U = torch.sum(torch.logical_or(y_pred[b] == part, y[b] == part))
            iou = I / float(U)
            part_ious.append(iou)
        batch_ious.append(torch.mean(torch.stack(part_ious))) # part IoU averaged
    return torch.mean(torch.stack(batch_ious))