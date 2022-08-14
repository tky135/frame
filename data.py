import os
import numpy as np
from torch.utils.data import Dataset
from util import readMNIST
from preprocess import split_train_val_test_csv
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import image
import torch
import json
import torchvision.transforms as T
from PIL import Image
import glob
################### Transforms #######################

train_augs = T.Compose([

    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    # T.ToPILImage(),
    # T.Resize(256),
    # T.Resize(224),
    T.ToTensor()])

test_augs = T.Compose([
    # T.ToPILImage(),
    T.Resize(227),
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

class ImgCls(Dataset):

    """
    Dataset for image classification


    """
    def __init__(self, partition, config) -> None:
        super().__init__()
        self.partition = partition
        self.config = config
        # root path of the dataset
        self.path = os.path.join("dataset", dataset)
        if partition == "inf":
            # x = pd.read_csv(os.path.join(self.path, "test.csv")).values
            # inf_path = os.listdir("dataset/lung")
            inf_path = glob.glob("dataset/lung/*/" + '*.png')
            self.x = np.array(inf_path)
            # self.x = np.array(["original\\or14.jpg"])
            self.y = None
        else:
            if not os.path.exists(os.path.join(self.path, partition + ".csv")):
                split_train_val_test_csv(self.path)
            df = pd.read_csv(os.path.join(self.path, partition + ".csv"))

            # convert string labels to ints
            # give one standard to label2int and int2label
            dict_file = os.path.join(os.path.join("experiments", config["exp_name"]), "dictionary.json")
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
    def __getitem__(self, index):

        if self.partition == "train":
            x = Image.open(os.path.join(self.path, self.x[index]))
            y = self.y[index]
            return train_augs(x), y
        elif self.partition == "test" or self.partition == "val":
            x = Image.open(os.path.join(self.path, self.x[index]))
            y = self.y[index]
            return test_augs(x), y
        elif self.partition == "inf":
            print(self.x[index])
            my_path = self.x[index] if self.path == self.x[index][:len(self.path)] else os.path.join(self.path, self.x[index])
            x = Image.open(my_path)
            return test_augs(x)
    def __len__(self):
        return self.x.shape[0]
        
    def get_mapping(self):
        dict_file = os.path.join(os.path.join("experiments", self.config["exp_name"]), "dictionary.json")
        if not os.path.exists(dict_file):
            raise Exception("dictionary.json file must be created by the train experiment")
        with open(dict_file, "r") as f:
            dictionary = json.load(f)
        return dictionary
        
class HuBMAP_HPA(Dataset):
    def __init__(self, data_folder) -> None:
        self.data_folder = os.path.join("dataset", data_folder)
    def preprocess(self):
        """
        1. Generate masks from rle in train_labels
        2. Create a csv all.csv with columns: id, img_path, label_path
        """ 
        train_img_path = os.path.join(self.data_folder, "train_images")
        train_label_path = os.path.join(self.data_folder, "train_labels")
        train_ann_path = os.path.join(self.data_folder, "train_annotations")
        if not os.path.exists(train_label_path):
            os.mkdir(train_label_path)
        fd = pd.read_csv(os.path.join(self.data_folder, "train.csv"))
        csv = open(os.path.join(self.data_folder, "all.csv"), 'w')
        csv.write("id, img_path, label_path\n")

        for i in range(fd.shape[0]):
            rle = fd["rle"][i].strip()
            height = fd["img_height"][i]
            width = fd["img_width"][i]
            id = fd["id"][i]
            print("processing", id)
            if not os.path.exists(os.path.join(train_label_path, "%d.npy" % id)):
                mask = self.rle2mask(rle, height, width)
                np.savetxt(os.path.join(train_label_path, "%d.npy" % id), mask)
                ### testing
                # new_rle = self.mask2rle(mask).strip()
                # assert(rle == new_rle)

            csv.write("%d, %s, %s\n" % (id, os.path.join("train_images", str(id) + ".tiff"), os.path.join("train_labels", str(id) + ".npy")))
    def rle2mask(self, rle: str, height: int, width: int) -> np.ndarray:
        """
        Convert label format from rle to 2D np.ndarray masks
        """
        mask = np.zeros((height * width,))
        rle_l = rle.strip().split(' ')
        for i in range(0, len(rle_l) - 1, 2):
            # print(i, "/", len(rle_l) - 1)
            start, step = int(rle_l[i]), int(rle_l[i + 1])
            # print(start, step)
            mask[start - 1: start - 1 + step] = 1
        mask = mask.reshape(width, height).T

        return mask
    def mask2rle(self, img: np.ndarray) -> str:
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels= img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)
    def visualize(self, idx: int = 0):
        train_img_path = os.path.join(self.data_folder, "train_images")
        train_label_path = os.path.join(self.data_folder, "train_labels")
        train_ann_path = os.path.join(self.data_folder, "train_annotations")
        df = pd.read_csv(os.path.join(self.data_folder, "train.csv"))
        img_id = df["id"][idx]

        my_img = np.array(Image.open(os.path.join(train_img_path, str(img_id) + ".tiff")))

        my_label = np.loadtxt(os.path.join(train_label_path, str(img_id) + ".npy")).astype(bool)
        my_ann = json.load(open(os.path.join(train_ann_path, str(img_id) + ".json")))
        for cycle in my_ann:
            for x,y in cycle:
                my_img[y-10:y+10,x-10:x+10, 0] = 255
        my_img[my_label, 2] = 255
        plt.imshow(my_img)
        plt.show()
if __name__ == "__main__":
    # dset = HuBMAP_HPA("organ")
    # dset.preprocess()
    files = glob.glob("dataset/lung/*/*/" + '*.png')
    print(files)
    # dset_train = ImgCls("train", None)
    # dset_test = ImgCls("test", None)
    # dset_val = ImgCls("val", None)
    # print(set(dset_train))
    # print(dset[3][0].shape)
    # print(dset[3][1])
    # # plt.imshow(dset[0][0].permute(1, 2, 0))
    # # plt.imshow(dset[1][0].permute(1, 2, 0))
    # # plt.imshow(dset[2][0].permute(1, 2, 0))
    # plt.imshow(dset[3][0].permute(1, 2, 0))
    # plt.show()
    