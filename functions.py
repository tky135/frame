import torch
import numpy as np
def CEloss(y_pred, y):
    # celoss = torch.nn.CrossEntropyLoss() # cross entropy includes softmax
    # y_pred = torch.softmax(y_orig, dim=1)
    # print(y_pred.shape)
    # print(y.squeeze().shape)
    # print(set(y.detach().cpu().numpy().flatten()))
    # raise Exception("break")
    # y[y == 255] = 21 ### Tesing

    ### MUST MAKE SURE THE SHAPES ARE CORRECT
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
    z = torch.sum((torch.max(y_pred, dim=1)[1] == y)) / y.flatten().shape[0]
    return z

# def seg_acc(y_pred, y):
#     z = torch.sum((torch.max(y_pred, dim=1)[1] == y)) / (y_pred.shape[0] * y_pred.shape[1] * y_pred.shape[2])
def acc_fn(y_pred, y):
    mse = torch.nn.MSELoss()
    return -torch.sqrt(mse(torch.log(y_pred.reshape(-1)), torch.log(y)))
    

# TODO 
def calculate_shape_IoU_np(pred_np, seg_np):
    # pred_np [B, N]        --predicted segmentation label for each point in a Batch
    # seg_np  [B, N]        --true segmentation label for each point in a Batch
    class_ious = []
    for b in range(seg_np.shape[0]):
        # get the set of classes
        print(np.unique(seg_np))
        for part in parts:
            # print(part)
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            # print(I, U)
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious)) # part IoU averaged
    return shape_ious