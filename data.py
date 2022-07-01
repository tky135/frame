import os
import numpy as np
from torch.utils.data import Dataset
from util import readMNIST
from preprocess import cvtFloatImg, procHouse, pred2l, split_train_val_test_csv
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import image
import torch
import json
import torchvision.transforms as T
from PIL import Image

################### Transforms #######################

train_augs = T.Compose([

    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    # T.ToPILImage(),
    # T.Resize(256),
    T.ToTensor()])

test_augs = T.Compose([
    # T.ToPILImage(),
    # T.Resize(1024),
    T.ToTensor()])

######################################################

dataset = "lung"

######################################################
class MNIST(Dataset):
    def __init__(self, partition):
        x = cvtFloatImg(readMNIST("dataset/FashionMNIST/raw/Fashion_Train"))
        y = readMNIST("dataset/FashionMNIST/raw/Fashion_Label")
        num = y.shape[0]
        if partition == "train":
            self.x = x[0:int(num * 9 / 10)]
            self.y = y[0:int(num * 9 / 10)]
        elif partition == "val":
            self.x = x[int(num * 9 / 10):]
            self.y = y[int(num * 9 / 10):]
        else:
            raise Exception("Wrong partition")

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]

class HousePrice(Dataset):
    def __init__(self, partition):
        self.partition = partition
        if partition == "train" or partition == "val":
            x = pd.read_csv("dataset/HousePrice/train.csv")
            x, y = pred2l("train")

            # randomly shuffle train set
            num = y.shape[0]
            indices = np.arange(num)
            new_indices = np.random.choice(indices, num, replace=False)
            x = x[new_indices]
            y = y[new_indices]

            if partition == "train":
                self.x = x[0:int(num * 9 / 10)]
                self.y = y[0:int(num * 9 / 10)]
            elif partition == "val":
                self.x = x[int(num * 9 / 10):]
                self.y = y[int(num * 9 / 10):]
        elif partition == "test":
            x = pd.read_csv("dataset/HousePrice/test.csv")
            self.x = pred2l("test")
            
        else:
            raise Exception("Not implemented")
    def __getitem__(self, index):
        if self.partition == "train" or self.partition == "val":
            return self.x[index], self.y[index]
        elif self.partition == "test":
            return self.x[index]
    def __len__(self):
        return self.x.shape[0]

# shared variable across
leaf_dict = [None]
class Leaf(Dataset):

    def __init__(self, partition):
        global leaf_dict
        super().__init__()
        self.path = os.path.join("dataset", "Leaf")
        self.partition = partition
        if partition == "train" or partition == "val":
            x = pd.read_csv(os.path.join(self.path, "train.csv"))

            # convert category strings into unique
            leaf_types = x["label"].unique()
            leaf_dict = dict(zip(leaf_types, range(len(leaf_types))))
            x = x.replace(leaf_dict)

            y = x["label"].values
            x = x["image"].values

            # randomly shuffle indices
            num = y.shape[0]
            indices = np.arange(num)
            np.random.seed(42)
            new_indices = np.random.choice(indices, num, replace=False)
            x = x[new_indices]
            y = y[new_indices]
            print(y[:10])

            if partition == "train":
                self.x = x[0:int(num * 9 / 10)]
                self.y = y[0:int(num * 9 / 10)]
            elif partition == "val":
                self.x = x[int(num * 9 / 10):]
                self.y = y[int(num * 9 / 10):]
            
            # read images into memory (too slow)
            # x = []
            # for i in range(self.x.shape[0]):
            #     x.append(torch.tensor(image.imread(os.path.join(self.path, self.x[i]))).unsqueeze(0))
            # self.x = torch.cat(x, axis=0)
            # print(self.x.shape)

        elif partition == "test":
            x = pd.read_csv(os.path.join(self.path, "train.csv"))

            # convert category strings into unique
            leaf_types = x["label"].unique()
            leaf_dict[0] = dict(zip(leaf_types, range(len(leaf_types))))
            x = pd.read_csv("dataset/Leaf/test.csv").values
            self.x = x.flatten()
            self.y = None
            
        else:
            raise Exception("Not implemented")

    def __getitem__(self, idx):
        x = Image.open(os.path.join(self.path, self.x[idx]))

        if self.partition == "train" or self.partition == "val":
            y = self.y[idx]
            return train_augs(x), y
        else:
            return test_augs(x)
    def __len__(self):
        return self.x.shape[0]

class ImgCls(Dataset):

    """
    Dataset for image classification


    """
    label2int = None
    int2label = None
    def __init__(self, partition, args) -> None:
        super().__init__()
        self.partition = partition

        # root path of the dataset
        self.path = os.path.join("dataset", dataset)
        if partition == "inf":
            # x = pd.read_csv(os.path.join(self.path, "test.csv")).values
            self.x = np.array(["orc30.png"])
            # self.x = np.array(["original\\or14.jpg"])
            self.y = None
        else:
            if not os.path.exists(os.path.join(self.path, partition + ".csv")):
                split_train_val_test_csv(self.path)
            df = pd.read_csv(os.path.join(self.path, partition + ".csv"))

            # convert string labels to ints
            # give one standard to label2int and int2label
            dict_file = os.path.join(os.path.join("outputs", args.exp_name), "dictionary.json")
            if partition == "train" and not os.path.exists(dict_file):
                cifar_types = df["label"].unique()
                dictionary = {"label2int" : dict(zip(cifar_types, range(len(cifar_types)))), "int2label" : dict(zip(range(len(cifar_types)), cifar_types))}
                with open(dict_file, "w") as f:
                    json.dump(dictionary, f)
            elif os.path.exists(dict_file):
                with open(dict_file, "r") as f:
                    dictionary = json.load(f)
            else:
                raise Exception("dictionary.json file must be created by the train experiment")
            
            label2int = dictionary["label2int"]
            df["label"] = df["label"].replace(label2int)

            self.y = df["label"].values
            self.x = df["image"].values

            # split train and val set
            # num = y.shape[0]
            # indices = np.arange(num)
            # np.random.seed(42) # I don't want to rely on this
            # new_indices = np.random.choice(indices, num, replace=False)
            # x = x[new_indices]
            # y = y[new_indices]

            # if partition == "train":
            #     self.x = x[0:int(num * 8 / 10)]
            #     self.y = y[0:int(num * 8 / 10)]
            # elif partition == "val":
            #     self.x = x[int(num * 8 / 10):int(num * 9 / 10)]
            #     self.y = y[int(num * 8 / 10):int(num * 9 / 10)]
            # elif partition == "test":
            #     self.x = x[int(num * 9 / 10):]
            #     self.y = y[int(num * 9 / 10):]

    def __getitem__(self, index):

        if self.partition == "train" or self.partition == "val" :
            x = Image.open(os.path.join(self.path, self.x[index]))
            y = self.y[index]
            return train_augs(x), y
        elif self.partition == "test":
            x = Image.open(os.path.join(self.path, self.x[index]))
            y = self.y[index]
            return test_augs(x), y
        elif self.partition == "inf":
            x = Image.open(os.path.join(self.path, self.x[index]))
            return test_augs(x)
    def __len__(self):
        return self.x.shape[0]

    def get_mapping(self):
        if self.label2int == None:
            x = pd.read_csv(os.path.join(self.path, "all.csv"))
            # convert category strings into unique
            cifar_types = x["label"].unique()
            ImgCls.label2int = dict(zip(cifar_types, range(len(cifar_types))))
            ImgCls.int2label = dict(zip(range(len(cifar_types)), cifar_types))
        return ImgCls.label2int, ImgCls.int2label
        

if __name__ == "__main__":
    dset = ImgCls("train")
    print(dset[3][0].shape)
    print(dset[3][1])
    # plt.imshow(dset[0][0].permute(1, 2, 0))
    # plt.imshow(dset[1][0].permute(1, 2, 0))
    # plt.imshow(dset[2][0].permute(1, 2, 0))
    plt.imshow(dset[3][0].permute(1, 2, 0))
    plt.show()
    