
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
from preprocess import *
from util import *
from data import *
from functions import acc_fn, RMSElog, CEloss, class_acc
import yaml
import datetime
from sklearn import metrics
############ VALIDATION ############
loss_fn = CEloss
acc_fn = class_acc
############ DATA SET ##############
DATASET = ImgSeg
############ MODEL #################
def get_model(config):
    return config["model"](n_category=22)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=False)
    # return model
####################################


def train(config, log):
    log.write("TRAIN\n\n")
    device = torch.device("cuda" if (config["cuda"] and torch.cuda.is_available()) else "cpu")
    print("Device: ", device)

    # set train dataloader
    train_loader = DataLoader(DATASET(partition="train", config=config), batch_size=config["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(DATASET(partition="val", config=config), batch_size=config["batch_size"], shuffle=False, drop_last=False)

    # set model

    model = get_model(config).to(device)
    # model = torch.nn.Sequential(nn.Linear(18, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    print(model)
    if device != torch.device("cpu"):
        model = nn.DataParallel(model)

    ## TODO add continue training
    if config["continue"] == True:
        model_path = "experiments/" + config["exp_name"] + "/model.t7"
        if not os.path.exists(model_path):
            raise Exception(model_path, "Does not exist")
        print("Loading state dict from: " + model_path + " ...")
        model.load_state_dict(torch.load(model_path))
    # set optimizer and scheduler
    # set different parameters for different layers

    # ResNet18
    # orig_para = [para for name, para in model.module.net.named_parameters() if name not in ['fc.weight', 'fc.bias']]
    # optimizer = optim.Adam([{'params': orig_para, 'lr': config["lr"]}, {'params': model.module.net.fc.parameters(), 'lr': config["lr"] * 10}], lr=config["lr"], weight_decay=config["weight_decay"])

    # AlexNet
    # orig_para = [para for name, para in model.module.net.named_parameters() if name not in ['classifier.6.weight', 'classifier.6.bias']]
    # optimizer = optim.Adam([{'params': orig_para, 'lr': config["lr"]}, {'params': model.module.net.classifier[6].parameters(), 'lr': config["lr"] * 10}], lr=config["lr"], weight_decay=config["weight_decay"])

    # General
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], momentum=config["momentum"])

    # decrease learning rate to 0.1 of itself at the end of training
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1 ** (1 / config["epochs"]))

    # set display and monitor list
    tr_loss_l = []
    tr_acc_l = []
    ev_loss_l = []
    ev_acc_l = []

    # set best model
    best_acc = -1e10
    
    for epoch in range(config["epochs"]):
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
            y_pred = model(x)['out']
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

        if not len(val_loader.dataset) == 0 or config["validate"]:
            avg_ev_loss, avg_ev_acc = val(config, log, model, val_loader)
            log.write("\tev_loss: " + "%.4f" % avg_ev_loss)
            log.write("\tev_acc: " + "%.4f" % avg_ev_acc)
            ev_loss_l.append(avg_ev_loss)
            ev_acc_l.append(avg_ev_acc)
            if avg_ev_acc >= best_acc:
                best_acc = avg_ev_acc
                path = ("experiments\\" + config["exp_name"] + "\\model.t7") if os.name == "nt" else ("experiments/" + config["exp_name"] + "/model.t7")
                torch.save(model.state_dict(), path)
                log.write("\tmodel saved. ")
            
        else:
            path = ("experiments\\" + config["exp_name"] + "\\model.t7") if os.name == "nt" else ("experiments/" + config["exp_name"] + "/model.t7")
            torch.save(model.state_dict(), path)
            log.write("model saved. ")

        # update best model
        log.write("\n")
        log.flush()

    # end of training
    fig = plt.figure()
    loss_plt = fig.add_subplot(121)
    acc_plt = fig.add_subplot(122)
    x = np.arange(config["epochs"])
    loss_plt.plot(x, tr_loss_l, 'r', label="train")
    loss_plt.plot(x, ev_loss_l, 'b', label="val")
    loss_plt.title.set_text("Loss wrt Epoch")
    loss_plt.legend()
    acc_plt.plot(x, tr_acc_l, 'r', label="train")
    acc_plt.plot(x, ev_acc_l, 'b', label="val")
    acc_plt.title.set_text("Accuracy wrt Epoch")
    acc_plt.legend()
    path = "experiments\\" + config["exp_name"] + "\\train.png" if os.name == "nt" else "experiments/" + config["exp_name"] + "/train.png"
    plt.savefig(path)
    plt.show()

    log.write("\nBest val acc: " + "%.4f" % best_acc + "\n")


def val(config, log, in_model, val_loader):
    device = torch.device("cuda" if (config["cuda"] and torch.cuda.is_available()) else "cpu")
    model = in_model
        # set val dataloader
    # val_loader = DataLoader(DATASET(partition="val", config=config), batch_size=config["batch_size"], shuffle=False)

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
            y_pred = model(x)['out']
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

def test(config):
    device = torch.device("cuda" if (config["cuda"] and torch.cuda.is_available()) else "cpu")

    model = get_model(config)
    if device != torch.device("cpu"):
        model = nn.DataParallel(model)
    model_path = "experiments/" + config["exp_name"] + "/model.t7"
    model.load_state_dict(torch.load(model_path))

    test_loader = DataLoader(DATASET(partition="test", config=config), batch_size=config["batch_size"], shuffle=False, drop_last=False)

    # set model to val
    model.eval()

    # set display and monitor list
    avg_ev_loss = 0
    avg_ev_acc = 0
    global_confusion_matrix = np.zeros((4, 4))
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

            # get confusion matrix
            confusion_matrix = metrics.confusion_matrix(y.cpu().numpy(), np.argmax(y_pred.cpu().numpy(), axis=1), labels=range(global_confusion_matrix.shape[0]))
            global_confusion_matrix += confusion_matrix
    avg_ev_loss /= len(test_loader.dataset)
    avg_ev_acc /= len(test_loader.dataset)
    int2label = test_loader.dataset.get_mapping()["int2label"]
    plot_confusion_matrix(global_confusion_matrix, [int2label[str(i)] for i in range(global_confusion_matrix.shape[0])])

    print("testing: ", avg_ev_loss, avg_ev_acc)
    return avg_ev_loss, avg_ev_acc


### TODO add displaying for different tasks

def inference(config):

    device = torch.device("cuda" if config["cuda"] else "cpu")
    model = get_model(config)
    if device != torch.device("cpu"):
        model = nn.DataParallel(model)
    model_path = "experiments/" + config["exp_name"] + "/model.t7"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    inf_dataset = DATASET(partition="inf", config=config)
    inf_loader = DataLoader(inf_dataset, batch_size=1, shuffle=False, drop_last=False)


    f = open("experiments/" + config["exp_name"] + "/test_result.csv", 'w')

    dictionary = inf_dataset.get_mapping()
    # raise Exception("break")
    # construct 
    with torch.no_grad():
        for x in inf_loader:
            x = x.to(device)
            y_pred = model(x)
            print(F.softmax(y_pred, dim=1))
            y_pred = torch.argmax(y_pred, dim=1)
            print(dictionary['int2label'])
            print("prediction", dictionary['int2label'][str(y_pred.item())])
            

    f.close()

if __name__ == "__main__":

    print("Reading configurations from config.yml...")
    with open("config.yml", 'r') as f:
        config = yaml.safe_load(f)
    print("Reading configurations from terminal...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--cuda", type=bool, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--exp_type", type=str, default=None)
    parser.add_argument("--validate", type=bool, default=None)
    parser.add_argument("--do_split", type=bool, default=None)
    parser.add_argument("--train_val_test_ratio", type=list, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("-f", "--force", type=str, default=None)
    parser.add_argument("--continue", type=bool, default=None)
    args = parser.parse_args()
    

    for arg_name in dir(args):
        if (arg_name[0] == '_' or not getattr(args, arg_name)):
            continue
        config[arg_name] = getattr(args, arg_name)
    # arguments that might require evaluation
    req_eval = ["lr", "weight_decay", "model"]
    for arg_name in req_eval:
        if type(arg_name) is not str:
            continue
        config[arg_name] = eval(config[arg_name])
    # generate default exp_name for lazy users
    if not config["exp_name"] or config["exp_name"] == "default":
        config["exp_name"] = config["model"].__name__ + "_" + config["dataset"] + "_" + str(datetime.date.today())
    print(config["exp_name"])
    # raise Exception("break")
    # try to split train val test
    # if config["do_split"]:
    #     if config["exp_type"] != "train":
    #         user = input("Your experiment type is: " + config["exp_type"] + ". Are you sure you want to do split? (y/n)")
    #         if user not in ["Y", "y"]:
    #             raise Exception("Canceled")
    #     split_train_val_test_csv(data_folder=os.path.join("/data", config["dataset"]), train_ratio=config["train_val_test_ratio"][0], val_ratio=config["train_val_test_ratio"][1], test_ratio=config["train_val_test_ratio"][2])
    if config["exp_type"] == "train":
        # make output directories

        # if exp_name already exists and is not default "exp" and force is not set, raise Exception
        if os.path.exists("experiments/" + config["exp_name"]) and config["exp_name"] != "exp" and not config["force"]:
            raise Exception("Already exist: " + "experiments/" + config["exp_name"])

        os.makedirs("experiments/" + config["exp_name"], exist_ok=True)
    
        # write initial log
        log = open("experiments/" + config["exp_name"] + "/log.txt", "a")
        log.write("\n")
        log.write(time.asctime())
        log.write("\n")

        # backup files

        if os.name == "nt":
            # if using windows
            os.system("copy *.py experiments\\" + config["exp_name"])
        else:
            os.system("cp *.py experiments/" + config["exp_name"])
        # reproducibility
        # torch.manual_seed(42)
        # np.random.seed(42)
        # torch.use_deterministic_algorithms(True)

    elif not os.path.exists(os.path.join("experiments", config["exp_name"])):
            raise Exception("Must train first")
    if config["exp_type"] == "inf":
        inference(config)
    elif config["exp_type"] == "test":
    # experiment
        test(config)
    elif config["exp_type"] == "val":
        val(config, None, None, True)
    else:
        train(config, log)