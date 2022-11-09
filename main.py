
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
from autoregress_model import *
from preprocess import *
from util import *
from data import *
from functions import *
import yaml
import datetime
from sklearn import metrics
import inspect
############ VALIDATION ############
loss_fn = CEloss

def train(config, log):
    log.write("TRAIN\n\n")
    device = torch.device("cuda" if (config["cuda"] and torch.cuda.is_available()) else "cpu")
    print("Device: ", device)

    # set train dataloader
    train_dataset = config["data"](partition="train", config=config)
    val_dataset = config["data"](partition="val", config=config)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=False, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False, num_workers=8)

    # set model
    model = config["model"](**config["task_data_arg"]).to(device)
    # model = torch.nn.Sequential(nn.Linear(18, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    # print(model)
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
    all_metrics = {"train": {}, "val": {}}

    # initialize all_metrics
    for metric in config["task_data_arg"]["train_metric_list"] + [config["task_data_arg"]["loss_fn"]]:
        all_metrics["train"][metric.__name__] = []
    for metric in config["task_data_arg"]["val_metric_list"] + [config["task_data_arg"]["loss_fn"]]:
        all_metrics["val"][metric.__name__] = []
    # set best model
    best_acc = -1e10
    
    for epoch in range(config["epochs"]):
        # new epoch
        log.write("Epoch " + str(epoch) + ": ")
        for metric in all_metrics["train"]:
            all_metrics["train"][metric].append(0)
        # epoch-wise averaged metrics
        # avg_metrics = {}

        # must support when unpacked values != 2
        for x_y in train_loader:
            # move to device
            x_y = [x_y[i].to(device) for i in range(len(x_y))]

            y_pred = model(x_y[0])

            loss = config["task_data_arg"]["loss_fn"](y_pred, *(x_y[1:]))

            all_metrics["train"][config["task_data_arg"]["loss_fn"].__name__][-1] += loss.item() * y.shape[0]
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate and print loss and other kinds of metrics
            print("Epoch: %d" % epoch, end='\t')
            print("loss: %.4f" % loss.item(), end='\t')
            with torch.no_grad():

                for metric in config["task_data_arg"]["train_metric_list"]:
                    acc = metric(y_pred, y)
                    print(metric.__name__ + ": %.4f" % acc, end='\t')
                    all_metrics["train"][metric.__name__][-1] += acc.item() * y.shape[0]
                # acc = acc_fn(y_pred, y)
            print()
        # end of an epoch

        # scheduler step
        scheduler.step()
        # val at the end of each epoch

        # log all metrics at each epoch
        for metric in all_metrics["train"]:
            all_metrics["train"][metric][-1] /= len(train_dataset)
            log.write(metric + ": %.4f\t" % all_metrics["train"][metric][-1])
        log.write("\n")
        # val at the end of each epoch
        if len(val_loader.dataset) > 0 and config["validate"]:
            val_avg_metrics = val(config, log, model, val_loader)
            for metric in val_avg_metrics:
                all_metrics["val"][metric].append(val_avg_metrics[metric])
            # save best model based on the first val metric
            _, cand = next(iter(val_avg_metrics.items()))
            if cand > best_acc:
                best_acc = cand
                path = ("experiments\\" + config["exp_name"] + "\\model.t7") if os.name == "nt" else ("experiments/" + config["exp_name"] + "/model.t7")
                torch.save(model.state_dict(), path)
                log.write("\tmodel saved. ")
            
        else:
            # if no validation, save model at the end of each epoch
            path = ("experiments\\" + config["exp_name"] + "\\model.t7") if os.name == "nt" else ("experiments/" + config["exp_name"] + "/model.t7")
            torch.save(model.state_dict(), path)
            log.write("model saved. ")

        # update best model
        log.write("\n")
        log.flush()

    # end of training

    fig = plt.figure()
    # combine train & val metrics into a single plot
    metric_plot = {}
    for metric in set(all_metrics["train"].keys()).union(set(all_metrics["val"].keys())):
        metric_plot[metric] = {}
        if metric in all_metrics["train"]:
            metric_plot[metric]["train"] = all_metrics["train"][metric]
        if metric in all_metrics["val"]:
            metric_plot[metric]["val"] = all_metrics["val"][metric]
    
    print(metric_plot.keys())
    # subplot
    x = np.arange(config["epochs"])
    n_rows = int(np.ceil(np.sqrt(len(metric_plot))))
    n_cols = int(np.ceil(len(metric_plot) / n_rows))
    count = 1
    for metric in metric_plot:
        ax = fig.add_subplot(n_rows, n_cols, count)
        count += 1
        ax.set_title(metric)
        for phase in metric_plot[metric]:
            ax.plot(x, metric_plot[metric][phase], label=phase)
        ax.legend()
    path = "experiments\\" + config["exp_name"] + "\\train.png" if os.name == "nt" else "experiments/" + config["exp_name"] + "/train.png"
    plt.savefig(path)
    plt.show()

    log.write("\nBest val acc: " + "%.4f" % best_acc + "\n")


def val(config, log, in_model, val_loader):
    # returns a dictionary of val metrics
    device = torch.device("cuda" if (config["cuda"] and torch.cuda.is_available()) else "cpu")
    model = in_model
        # set val dataloader
    # val_loader = DataLoader(config["task"](partition="val", config=config), batch_size=config["batch_size"], shuffle=False)

    # set model to val
    model.eval()

    # set display and monitor list
    avg_metrics = {}

    with torch.no_grad():
        for x, y in val_loader:
            # move to device
            x = x.to(device)
            y = y.to(device)

            # forward pass
            y_pred = model(x)
            for metric in config["task_data_arg"]["val_metric_list"] + [config["task_data_arg"]["loss_fn"]]:
                acc = metric(y_pred, y)
                if metric.__name__ not in avg_metrics:
                    avg_metrics[metric.__name__] = acc.item() * y.shape[0]
                else:
                    avg_metrics[metric.__name__] += acc.item() * y.shape[0]

    for key in avg_metrics:
        avg_metrics[key] /= len(val_loader.dataset)
    # set model back to train
    model.train()
    # return valuation loss and other metrics
    print("validation: \t")
    for key in avg_metrics:
        print(key + ": %.4f" % avg_metrics[key], end='\t')
        log.write("\t" + key + ": %.4f" % avg_metrics[key])
    print()
    return avg_metrics

# def test(config):
#     device = torch.device("cuda" if (config["cuda"] and torch.cuda.is_available()) else "cpu")

#     model = config["model"](**config["task_data_arg"])
#     if device != torch.device("cpu"):
#         model = nn.DataParallel(model)
#     model_path = "experiments/" + config["exp_name"] + "/model.t7"
#     model.load_state_dict(torch.load(model_path))

#     test_loader = DataLoader(config["task"](partition="test", config=config), batch_size=config["batch_size"], shuffle=False, drop_last=False)

#     # set model to val
#     model.eval()

#     # set display and monitor list
#     avg_ev_loss = 0
#     avg_ev_acc = 0
#     global_confusion_matrix = np.zeros((4, 4))
#     with torch.no_grad():

#         for x, y in test_loader:
#             # move to device
#             x = x.to(device)
#             y = y.to(device)

#             # forward pass
#             y_pred = model(x)
#             loss = loss_fn(y_pred, y)
#             acc = acc_fn(y_pred, y)
#             avg_ev_loss += loss.item() * y.shape[0]
#             avg_ev_acc += acc.item() * y.shape[0]

#             # get confusion matrix
#             confusion_matrix = metrics.confusion_matrix(y.cpu().numpy(), np.argmax(y_pred.cpu().numpy(), axis=1), labels=range(global_confusion_matrix.shape[0]))
#             global_confusion_matrix += confusion_matrix
#     avg_ev_loss /= len(test_loader.dataset)
#     avg_ev_acc /= len(test_loader.dataset)
#     int2label = test_loader.dataset.get_mapping()["int2label"]
#     plot_confusion_matrix(global_confusion_matrix, [int2label[str(i)] for i in range(global_confusion_matrix.shape[0])])

#     print("testing: ", avg_ev_loss, avg_ev_acc)
#     return avg_ev_loss, avg_ev_acc


### TODO add displaying for different tasks

# def inference(config):

#     device = torch.device("cuda" if config["cuda"] else "cpu")
#     model = config["model"](**config["task_data_arg"])
#     if device != torch.device("cpu"):
#         model = nn.DataParallel(model)
#     model_path = "experiments/" + config["exp_name"] + "/model.t7"
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     inf_dataset = config["task"](partition="inf", config=config)
#     inf_loader = DataLoader(inf_dataset, batch_size=1, shuffle=False, drop_last=False)


#     f = open("experiments/" + config["exp_name"] + "/test_result.csv", 'w')

#     dictionary = inf_dataset.get_mapping()
#     # raise Exception("break")
#     # construct 
#     with torch.no_grad():
#         for x in inf_loader:
#             x = x.to(device)
#             y_pred = model(x)
#             print(F.softmax(y_pred, dim=1))
#             y_pred = torch.argmax(y_pred, dim=1)
#             print(dictionary['int2label'])
#             print("prediction", dictionary['int2label'][str(y_pred.item())])
            

#     f.close()

if __name__ == "__main__":

    # read config.yml as a dictionary
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
    parser.add_argument("-f", "--force", type=str, default=None)
    parser.add_argument("--continue", type=bool, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataroot", type=str, default=None)
    args = parser.parse_args()
    
    # overwrite config if provided in command line argument
    for arg_name in dir(args):
        if (arg_name[0] == '_' or not getattr(args, arg_name)):
            continue
        config[arg_name] = getattr(args, arg_name)
    # arguments that might require first level evaluation
    req_eval_1 = ["lr", "weight_decay", "model", "data"]
    for arg_name in req_eval_1:
        # if config[arg_name] is not read in as a string, continue
        if type(config[arg_name]) is not str:
            continue
        config[arg_name] = eval(config[arg_name])

    # arguments that might require second level evaluation
    # req_eval_2 = ["metric_list"]
    # for arg_name in req_eval_2:
    #     new_list = []
    #     for item in config[arg_name]:
    #         if type(item) is str:
    #             new_list.append(eval(item))
    #         else:
    #             new_list.append(item)
    #     config[arg_name] = new_list
    # generate default exp_name for lazy users
    if not config["exp_name"] or config["exp_name"] == "default":
        config["exp_name"] = config["model"].__name__ + "_" + config["data"].__name__ + "_" + str(datetime.date.today())
    print("Experiment name: ", config["exp_name"])
    # generate arguments to model

    # (inspect.isfunction(m[1]) or 
    config["task_data_arg"] = dict([m for m in inspect.getmembers(config['data']) if not (inspect.ismethod(m[1]) or m[0].startswith('_'))])     # risky, but let's keep it this way for now
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