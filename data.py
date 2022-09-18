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
import trimesh
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
    # T.Resize(227),
    T.ToTensor()])

######################################################

# dataset = "lung"

######################################################
# class MNIST(Dataset):
#     def __init__(self, partition):
#         x = cvtFloatImg(readMNIST("dataset/FashionMNIST/raw/Fashion_Train"))
#         y = readMNIST("dataset/FashionMNIST/raw/Fashion_Label")
#         num = y.shape[0]
#         if partition == "train":
#             self.x = x[0:int(num * 9 / 10)]
#             self.y = y[0:int(num * 9 / 10)]
#         elif partition == "val":
#             self.x = x[int(num * 9 / 10):]
#             self.y = y[int(num * 9 / 10):]
#         else:
#             raise Exception("Wrong partition")

#     def __getitem__(self, index):
#         return self.x[index], self.y[index]

#     def __len__(self):
#         return self.y.shape[0]

# class HousePrice(Dataset):
#     def __init__(self, partition):
#         self.partition = partition
#         if partition == "train" or partition == "val":
#             x = pd.read_csv("dataset/HousePrice/train.csv")
#             x, y = pred2l("train")

#             # randomly shuffle train set
#             num = y.shape[0]
#             indices = np.arange(num)
#             new_indices = np.random.choice(indices, num, replace=False)
#             x = x[new_indices]
#             y = y[new_indices]

#             if partition == "train":
#                 self.x = x[0:int(num * 9 / 10)]
#                 self.y = y[0:int(num * 9 / 10)]
#             elif partition == "val":
#                 self.x = x[int(num * 9 / 10):]
#                 self.y = y[int(num * 9 / 10):]
#         elif partition == "test":
#             x = pd.read_csv("dataset/HousePrice/test.csv")
#             self.x = pred2l("test")
            
#         else:
#             raise Exception("Not implemented")
#     def __getitem__(self, index):
#         if self.partition == "train" or self.partition == "val":
#             return self.x[index], self.y[index]
#         elif self.partition == "test":
#             return self.x[index]
#     def __len__(self):
#         return self.x.shape[0]

class ImgCls(Dataset):

    """
    Dataset for image classification


    """
    def __init__(self, partition, config) -> None:
        super().__init__()
        self.partition = partition
        self.config = config
        # root path of the dataset
        self.path = os.path.join("/data", config["dataset"])



        if partition == "inf":
            # x = pd.read_csv(os.path.join(self.path, "test.csv")).values
            self.x = np.array(os.listdir("/data/lung"))
            # self.x = np.array(["original\\or14.jpg"])
            self.y = None
        else:
            if not os.path.exists(os.path.join(self.path, partition + ".csv")):
                user = input(os.path.join(self.path, partition + ".csv") + " does not exist, do split?(May overwrite other existing csv)(y/n)")
                if user not in ["y", "Y"]:
                    raise Exception("Canceled")
                split_train_val_test_csv(self.path, config["train_val_test_ratio"][0], config["train_val_test_ratio"][1], config["train_val_test_ratio"][2])
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


class PCCls(Dataset):
    def __init__(self, partition, config) -> None:
        super().__init__()
        self.partition = partition
        self.config = config
        # root path of the dataset
        self.path = os.path.join("/data", config["dataset"])

        ## TODO
        if partition == "inf":
            self.x = None
            self.y = None

        else:
            # if no csv for partition exist, try split
            if not os.path.exists(os.path.join(self.path, partition + ".csv")):
                user = input(os.path.join(self.path, partition + ".csv") + " does not exist, do split?(May overwrite other existing csv)(y/n)")
                if user not in ["y", "Y"]:
                    raise Exception("Canceled")
                split_train_val_test_csv(self.path, config["train_val_test_ratio"][0], config["train_val_test_ratio"][1], config["train_val_test_ratio"][2])
            # if so, read a dataframe
            df = pd.read_csv(os.path.join(self.path, partition + ".csv"))

        
            ## csv: meshfile, class

            # convert string labels to ints
            # give one standard to label2int and int2label
            dict_file = os.path.join(os.path.join("experiments", config["exp_name"]), "dictionary.json")
            # if train, write dict_file
            if partition == "train" and not os.path.exists(dict_file):
                cifar_types = df["y"].unique()
                dictionary = {"label2int" : dict(zip(cifar_types, range(len(cifar_types)))), "int2label" : dict(zip(range(len(cifar_types)), cifar_types))}
                with open(dict_file, "w") as f:
                    json.dump(dictionary, f)
            elif os.path.exists(dict_file):
                with open(dict_file, "r") as f:
                    dictionary = json.load(f)
            else:
                raise Exception("dictionary.json file must bImgClse created by the train experiment")

            label2int = dictionary["label2int"]
            df["y"] = df["y"].replace(label2int)
            self.y = df["y"].values
            self.x = df["x"].values
    def __getitem__(self, index):
        if self.partition == "train":
            mesh = trimesh.load(file_obj=open(os.path.join(self.path, self.x[index])), file_type="off")
            pc = trimesh.sample.sample_surface(mesh, self.config["num_samples"])[0]
            pc = self.normalize(pc)
            return torch.tensor(pc, dtype=torch.float32), self.y[index]
    def __len__(self):
        return self.x.shape[0]

    def index_points(self, indices, values):
        pass
    def normalize(self, pc):
        """
        np array
        """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def visualize(self, pc, label=None):
        """
        Visualize a point cloud, generate a .obj file (for now)
        """
        f = open("pc.obj", "w")
        for i in range(pc.shape[0]):
            f.write("v " + str(float(pc[i, 0])) + " " + str(float(pc[i, 1])) + " " + str(float(pc[i, 2])) + "\n")
        f.close()
class HuBMAP_HPA(Dataset):
    def __init__(self, data_folder) -> None:
        self.data_folder = os.path.join("/data", data_folder)
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
    pcds = PCCls("train", {"dataset":"ModelNet10", "train_val_test_ratio":[0.8, 0.1, 0.1], "exp_name":"test", "num_samples":4096})
    pcds.visualize(pcds[41][0])
    print(pcds[41][1])
    
# Notes:
# ModelNet and shapenet both are artificial mesh datasets. Why not learn on mesh directly
# self-supervised: predict new point's line, surface, corner: how human recognize things. Think of BERT