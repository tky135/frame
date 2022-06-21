import torch

def CEloss(y_orig, y):
    # print(y_pred.shape)
    # print(y.shape)
    # print(y_pred)
    # print(y)
    # print(y_pred.shape)
    # print(y.shape)
    # raise Exception("break")
    celoss = torch.nn.CrossEntropyLoss() # cross entropy includes softmax
    y_pred = torch.softmax(y_orig, dim=1)
    z = celoss(y_orig, y)
    # z_mid = - torch.log(y_pred[torch.arange(y_pred.shape[0]), y.type(torch.int64)])
    # z = - y_pred[torch.arange(y_pred.shape[0]), y.type(torch.int64)]
    # z = torch.sum(z_mid)
    # z /= y_pred.shape[0]
    if z == torch.inf or z > 1e20:
        print(y_orig)
        print(y_pred)
        print(y)
        # print(z_mid)
        raise Exception("infinity error")

    return z

def RMSElog(y_pred, y):
    # print(y_pred.shape)
    # print(y.shape)
    # print(y_pred)
    # print(y)
    y_pred = torch.clamp(y_pred, 1e-10, float('inf'))
    # print(y_pred)
    mse = torch.nn.MSELoss()
    # return torch.mean(1/2 * (y_pred - y) ** 2)
    # return torch.mean(torch.sqrt((torch.log(y_pred.reshape(-1)) - torch.log(y))**2 / 2))
    return torch.sqrt(mse(torch.log(y_pred.reshape(-1)), torch.log(y)))

def class_acc(y_pred, y):
    z = torch.sum((torch.max(y_pred, dim=1)[1] == y)) / y_pred.shape[0]
    return z
def acc_fn(y_pred, y):
    mse = torch.nn.MSELoss()
    return -torch.sqrt(mse(torch.log(y_pred.reshape(-1)), torch.log(y)))
    
    return z

