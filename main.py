
import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import *
from data import MNIST, HousePrice, ImgCls, leaf_dict
from functions import acc_fn, RMSElog, CEloss, class_acc

############ VALIDATION ############
loss_fn = CEloss
acc_fn = class_acc
############ DATA SET ##############
DATASET = ImgCls
############ MODEL #################
def get_model():
    return ResNet18(input_channels=3, n_category=4, pretrained=True)
####################################


def train(args, log):
    log.write("TRAIN\n\n")
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print(device)

    # set train dataloader
    train_loader = DataLoader(DATASET(partition="train", args=args), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(DATASET(partition="val", args=args), batch_size=args.batch_size, shuffle=False, drop_last=False)

    # set model

    model = get_model()
    # model = torch.nn.Sequential(nn.Linear(18, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    if device != torch.device("cpu"):
        model = nn.DataParallel(model)
    print(model)
    # set optimizer and scheduler
    # set different parameters for different layers

    # ResNet18
    # orig_para = [para for name, para in model.module.net.named_parameters() if name not in ['fc.weight', 'fc.bias']]
    # optimizer = optim.Adam([{'params': orig_para, 'lr': args.lr}, {'params': model.module.net.fc.parameters(), 'lr': args.lr * 10}], lr=args.lr, weight_decay=args.weight_decay)

    # AlexNet
    # orig_para = [para for name, para in model.module.net.named_parameters() if name not in ['classifier.6.weight', 'classifier.6.bias']]
    # optimizer = optim.Adam([{'params': orig_para, 'lr': args.lr}, {'params': model.module.net.classifier[6].parameters(), 'lr': args.lr * 10}], lr=args.lr, weight_decay=args.weight_decay)

    # General
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    # decrease learning rate to 0.1 of itself at the end of training
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1 ** (1 / args.epochs))

    # set display and monitor list
    tr_loss_l = []
    tr_acc_l = []
    ev_loss_l = []
    ev_acc_l = []

    # set best model
    best_acc = -1e10
    
    for epoch in range(args.epochs):
        # new epoch
        log.write("Epoch " + str(epoch) + ": ")

        # epoch-wise averaged metrics
        avg_tr_loss = 0
        avg_tr_acc = 0
        avg_ev_loss = 0
        avg_ev_acc = 0

        for x, y in train_loader:
            # move to device
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate other kinds of metrics
            with torch.no_grad():
                acc = acc_fn(y_pred, y)
            
            avg_tr_loss += loss.item() * y.shape[0]
            avg_tr_acc += acc.item() * y.shape[0]
            print("Epoch: %d" % epoch, end='\t')
            print("loss: %.4f" % loss.item(), end='\t')
            print("acc: %.4f" % acc.item(), end='\t')
            print()

        # end of an epoch

        # scheduler step
        scheduler.step()
        # val at the end of each epoch
        avg_tr_acc /= len(train_loader.dataset)
        avg_tr_loss /= len(train_loader.dataset)

        # log all metrics at each epoch
        log.write("\ttr_loss: " + "%.4f" % avg_tr_loss)
        log.write("\ttr_acc: " + "%.4f" % avg_tr_acc)

        # add metrics to global list for displaying
        tr_loss_l.append(avg_tr_loss)
        tr_acc_l.append(avg_tr_acc)

        if not len(val_loader.dataset) == 0 or not args.no_val:
            avg_ev_loss, avg_ev_acc = val(args, log, model, val_loader)
            log.write("\tev_loss: " + "%.4f" % avg_ev_loss)
            log.write("\tev_acc: " + "%.4f" % avg_ev_acc)
            ev_loss_l.append(avg_ev_loss)
            ev_acc_l.append(avg_ev_acc)
            if avg_ev_acc > best_acc:
                best_acc = avg_ev_acc
                path = ("outputs\\" + args.exp_name + "\\model.t7") if os.name == "nt" else ("outputs/" + args.exp_name + "/model.t7")
                torch.save(model.state_dict(), path)
                log.write("model saved. ")
        else:
            path = ("outputs\\" + args.exp_name + "\\model.t7") if os.name == "nt" else ("outputs/" + args.exp_name + "/model.t7")
            torch.save(model.state_dict(), path)
            log.write("model saved. ")

        # update best model
        log.write("\n")
        log.flush()

    # end of training
    fig = plt.figure()
    loss_plt = fig.add_subplot(121)
    acc_plt = fig.add_subplot(122)
    x = np.arange(args.epochs)
    loss_plt.plot(x, tr_loss_l, 'r', label="train")
    loss_plt.plot(x, ev_loss_l, 'b', label="val")
    loss_plt.title.set_text("Loss wrt Epoch")
    loss_plt.legend()
    acc_plt.plot(x, tr_acc_l, 'r', label="train")
    acc_plt.plot(x, ev_acc_l, 'b', label="val")
    acc_plt.title.set_text("Accuracy wrt Epoch")
    acc_plt.legend()
    path = "outputs\\" + args.exp_name + "\\train.png" if os.name == "nt" else "outputs/" + args.exp_name + "/train.png"
    plt.savefig(path)
    plt.show()

    log.write("\nBest val acc: " + "%.4f" % best_acc + "\n")


def val(args, log, in_model, val_loader):
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    model = in_model
        # set val dataloader
    # val_loader = DataLoader(DATASET(partition="val", args=args), batch_size=args.batch_size, shuffle=False)

    # set model to val
    model.eval()

    # set display and monitor list
    avg_ev_loss = 0
    avg_ev_acc = 0

    with torch.no_grad():
        for x, y in val_loader:
            # move to device
            x = x.to(device)
            y = y.to(device)

            # forward pass
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            acc = acc_fn(y_pred, y)
            
            avg_ev_loss += loss.item() * y.shape[0]
            avg_ev_acc += acc.item() * y.shape[0]

    avg_ev_loss /= len(val_loader.dataset)
    avg_ev_acc /= len(val_loader.dataset)
    # set model back to train
    model.train()
    # return valuation loss and other metrics
    print("validation: ", avg_ev_loss, avg_ev_acc)
    return avg_ev_loss, avg_ev_acc


def test(args):
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

    model = get_model()
    if device != torch.device("cpu"):
        model = nn.DataParallel(model)
    model_path = "outputs/" + args.exp_name + "/model.t7"
    model.load_state_dict(torch.load(model_path))

        # set val dataloader
    test_loader = DataLoader(DATASET(partition="test", args=args), batch_size=args.batch_size, shuffle=False)

    # set model to val
    model.eval()

    # set display and monitor list
    avg_ev_loss = 0
    avg_ev_acc = 0

    with torch.no_grad():
        for x, y in test_loader:
            # move to device
            x = x.to(device)
            y = y.to(device)

            # forward pass
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            acc = acc_fn(y_pred, y)
            
            avg_ev_loss += loss.item() * y.shape[0]
            avg_ev_acc += acc.item() * y.shape[0]

    avg_ev_loss /= len(test_loader.dataset)
    avg_ev_acc /= len(test_loader.dataset)
    # set model back to train
    model.train()
    # return valuation loss and other metrics
    print("testing: ", avg_ev_loss, avg_ev_acc)
    return avg_ev_loss, avg_ev_acc



def inference(args):

    device = torch.device("cuda" if args.cuda else "cpu")
    model = get_model()
    if device != torch.device("cpu"):
        model = nn.DataParallel(model)
    model_path = "outputs/" + args.exp_name + "/model.t7"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    inf_dataset = DATASET(partition="inf", args=args)
    inf_loader = DataLoader(inf_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)


    f = open("outputs/" + args.exp_name + "/test_result.csv", 'w')

    _, inverse_dict = inf_dataset.get_mapping()
    # raise Exception("break")
    # construct 
    with torch.no_grad():
        count = 18353
        for x in inf_loader:
            x = x.to(device)
            y_pred = model(x)
            print(y_pred)

            y_pred = torch.argmax(y_pred, dim=1)
            # # print(y_pred.shape)
            # raise Exception("break")
            for i in y_pred.detach().cpu().flatten():
                f.write("images/" + str(count) + ".jpg")
                f.write(",")
                f.write(str(inverse_dict[i.item()]))
                f.write("\n")
                count += 1

    f.close()



    

if __name__ == "__main__":

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--exp_type", type=str, default="train")
    parser.add_argument("--no_val", type=bool, default=False)
    args = parser.parse_args()


    if args.exp_type == "train":
        # make output directories
        if os.path.exists("outputs/" + args.exp_name) and args.exp_name != "exp":
            raise Exception("Already exist")

        os.makedirs("outputs/" + args.exp_name, exist_ok=True)
        
        # write initial log
        log = open("outputs/" + args.exp_name + "/log.txt", "a")
        log.write("\n")
        log.write(time.asctime())
        log.write("\n")

        # backup files

        if os.name == "nt":
            # if using windows
            os.system("copy *.py outputs\\" + args.exp_name)
        else:
            os.system("cp *.py outputs/" + args.exp_name)
        # reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        # torch.use_deterministic_algorithms(True)
    elif not os.path.exists(os.path.join("outputs", args.exp_name)):
            raise Exception("Must train first")
    if args.exp_type == "inf":
        inference(args)
    elif args.exp_type == "test":
    # experiment
        test(args)
    elif args.exp_type == "val":
        val(args, None, None, True)
    else:
        train(args, log)

