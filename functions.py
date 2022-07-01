import torch

def CEloss(y_orig, y):
    celoss = torch.nn.CrossEntropyLoss() # cross entropy includes softmax
    # y_pred = torch.softmax(y_orig, dim=1)
    z = celoss(y_orig, y)
    if z == torch.inf:
        print("y_orig: ")
        print(y_orig)
        print("y_pred: ")
        # print(y_pred)
        print(y)
        # print(z_mid)
        raise Exception("infinity error")

    return z

def RMSElog(y_pred, y):
    y_pred = torch.clamp(y_pred, 1e-10, float('inf'))
    mse = torch.nn.MSELoss()
    return torch.sqrt(mse(torch.log(y_pred.reshape(-1)), torch.log(y)))

def class_acc(y_pred, y):
    z = torch.sum((torch.max(y_pred, dim=1)[1] == y)) / y_pred.shape[0]
    return z
def acc_fn(y_pred, y):
    mse = torch.nn.MSELoss()
    return -torch.sqrt(mse(torch.log(y_pred.reshape(-1)), torch.log(y)))
    