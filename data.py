import os
import numpy as np
# from util import readMNIST
from preprocess import split_train_val_test_csv
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import image
import torch
import json
import torchvision.transforms as T
import torchvision
from PIL import Image
import glob
import trimesh
import warnings
from tqdm import tqdm
import sys
from torch import Tensor
from typing import Tuple, List
from util import read_img, write_img
from functions import *
from torch.utils.data import Dataset
### Level 0: base dataset class
class csvDataset(Dataset):
    def __init__(self, partition, config) -> None:
        super().__init__()

        # initialize (ok)
        self.partition = partition
        self.config = config
        if "data_path" in config["task_data_arg"] and config["task_data_arg"]["data_path"] is not None and os.path.isdir(config["task_data_arg"]["data_path"]):
            self.path = config["task_data_arg"]["data_path"]
        else:
            self.path = os.path.join(config["dataroot"], config["data"].__name__)
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
        self.dict_store = os.path.join(self.path, "dict_store.json")
        self.dict = {}

        # loading and dumping dict_store (ok)
        # user can manually provide a dictionary file
        if os.path.exists(self.dict_store):
            self.dict = json.load(open(self.dict_store))

        ### TODO single out inf ??? 
        if partition == "inf":
            self.x = None
            self.y = None
        else:
            if config["do_split"] == True or not os.path.exists(os.path.join(self.path, partition + ".csv")):
                print(os.path.join(self.path, partition + ".csv") + " does not exist, do split?(May overwrite other existing csv)(y/n)", file=sys.stderr)
                user = input()
                
                if user not in ["y", "Y"]:
                    raise Exception("Canceled")
                self._split_train_val_test_csv()
                config["do_split"] = False  # only do_split once in one experiment
            df = pd.read_csv(os.path.join(self.path, partition + ".csv"))

            self.x = df["x"].values
            self.y = df["y"].values

        # store dictionary to dict_store
        # if not os.path.exists(self.dict_store):
        json.dump(self.dict, open(self.dict_store, "w"))
    def __getitem__(self, index):
        # not considering partition for now
        return self.read_xy(self.x[index], self.y[index])
        # when testing or inferencing, now sure what to do


    def __len__(self):
        return len(self.y)
    def _split_train_val_test_csv(self) -> None:
        """
        This should be a function of class dataset
        """
        train_ratio, val_ratio, test_ratio = self.config["train_val_test_ratio"]
        if train_ratio + val_ratio + test_ratio != 1:
            raise Exception("train ratio + val ratio + test ratio should be 1")
        self._generate_all_csv()

        all_df = pd.read_csv(os.path.join(self.path, "all.csv"))
        all_df = all_df.sample(frac=1).reset_index(drop=True)
        # print(all_df.info)
        length = all_df.shape[0]
        all_df.iloc[:int(train_ratio * length), :].to_csv(os.path.join(self.path, "train.csv"), index=False)
        all_df.iloc[int(train_ratio * length) : int((train_ratio + val_ratio) * length), :].to_csv(os.path.join(self.path, "val.csv"), index=False)
        all_df.iloc[int((train_ratio + val_ratio) * length): , :].to_csv(os.path.join(self.path, "test.csv"), index=False)
    
    def _generate_all_csv(self):
        """
        This should be written by user
        """
        allcsv = open(os.path.join(self.path, "all.csv"), "w")
        allcsv.write("x,y\n")
        x_list, y_list = self.get_all_xy_and_preprocess()
        for x, y in zip(x_list, y_list):
            allcsv.write(str(x) + "," + str(y) + "\n")

    def get_all_xy_and_preprocess(self):
        ### MUST IMPLEMENT
        raise Exception("Not Implemented")
    def preprocess(self, *args):
        raise Exception("Not Implemented")
    def read_xy(self, x, y):
        ### MUST IMPLEMENT
        raise Exception("Not Implemented")
    def train_augs(self, x, y):
        raise Exception("Not Implemented")
    def test_augs(self, x, y):
        raise Exception("Not Implemented")

### Level 1: experiment type
class AutoRegress(csvDataset):
    train_metric_list = []
    val_metric_list = []
    loss_fn = neg_log_likelihood
    n_values = 20
    def __init__(self, partition, config) -> None:
        super().__init__(partition, config)
    def get_all_xy_and_preprocess(self):
        count = 5000
        rand = np.random.RandomState(0)
        samples = 0.4 + 0.1 * rand.randn(count)
        data = np.digitize(samples, np.linspace(0.0, 1.0, 20))
        return data, np.ones_like(data)
    def read_xy(self, x, y):
        return torch.tensor(x), torch.tensor(y)
        
class ImgCls(csvDataset):
    def __init__(self, partition, config) -> None:
        super().__init__(partition, config)

    def get_all_xy_and_preprocess(self):
        X, Y = [], []
        if "_counter" not in self.dict:
            self.dict["_counter"] = 0
        for folder in os.listdir(self.path):
            ext_folder = os.path.join(self.path, folder)
            if not os.path.isdir(ext_folder):
                continue
            if folder not in self.dict:
                self.dict[folder] = self.dict["_counter"]
                self.dict["_counter"] += 1
            for img in os.listdir(ext_folder):
                ext_img = os.path.join(ext_folder, img)
                X.append(ext_img)
                Y.append(self.dict[folder])
        return X, Y
    def read_xy(self, x, y):
        # for train, val and test
        # print(set((read_img(x) / 255).flatten().numpy()))
        # print(read_img(x).shape)
        # raise Exception("break")
        return self.aug_resize(read_img(x)), y
    def aug_resize(self, x):
        # Resize the image
        return T.functional.resize(x, (224, 224))

class ImgSeg(csvDataset):
    train_metric_list = [class_acc, calculate_shape_IoU_np]
    val_metric_list = [calculate_shape_IoU_np, class_acc]
    loss_fn = CEloss

    # copied & modified from functional_tensor.py
    def _pad_symmetric(self, img: Tensor, padding: List[int]) -> Tensor:
        """
        pad: List[int]. 
                If len is 2, expect [y_dir, x_dir]. 
                If len is 4, expect [left, right, top, down]
        """
        # padding is left, right, top, bottom
        if len(padding) == 2:
            # if padding len is 2, expect: [y direction, x direction]
            padding = [padding[1], padding[1], padding[0], padding[0]]
        in_sizes = img.size()

        _x_indices = [i for i in range(in_sizes[-1])]  # [0, 1, 2, 3, ...]
        left_indices = [i for i in range(padding[0] - 1, -1, -1)]  # e.g. [3, 2, 1, 0]
        right_indices = [-(i + 1) for i in range(padding[1])]  # e.g. [-1, -2, -3]
        x_indices = torch.tensor(left_indices + _x_indices + right_indices, device=img.device)

        _y_indices = [i for i in range(in_sizes[-2])]
        top_indices = [i for i in range(padding[2] - 1, -1, -1)]
        bottom_indices = [-(i + 1) for i in range(padding[3])]
        y_indices = torch.tensor(top_indices + _y_indices + bottom_indices, device=img.device)

        ndim = img.ndim
        if ndim == 3:
            return img[:, y_indices[:, None], x_indices[None, :]]
        elif ndim == 4:
            return img[:, :, y_indices[:, None], x_indices[None, :]]
        elif ndim == 2:
            return img[y_indices[:, None], x_indices[None, :]]
        else:
            raise RuntimeError("Symmetric padding of N-D tensors are not supported yet")
    def _random_crop(self, x: Tensor, y: Tensor, shape: Tuple[int, int]):
        """
        Random cropping for sementic segmentation. If input shape is less than desired crop shape, symmetric, even padding is applied. img and labels will be cropped (and padded) in the same way. 
        Input: 
        x: Tensor with shape [3, h, w]
        y: Tensor with shape [h, w] or [1, h, w]
        shape: Tuple[int, int] Desired shape with [h, w]
        
        """
        y = y.squeeze()
        assert x.shape[1] == y.shape[0] and x.shape[2] == y.shape[1]
        pad = torch.tensor(shape) - torch.tensor(x.shape[1:])
        pad[pad < 0] = 0
        pad = torch.div(pad + 1, 2, rounding_mode="floor")
        if torch.max(pad) > 0:
            # print("before padding: ", x.shape, y.shape)
            pad = list(pad.numpy())
            # print(pad)
            x, y = self._pad_symmetric(x, pad), self._pad_symmetric(y, pad)
        rect = T.RandomCrop.get_params(x, shape)
        return T.functional.crop(x, *rect), T.functional.crop(y, *rect)
    def __init__(self, partition, config) -> None:
        super().__init__(partition, config)


    def aug_crop(self, x, y):
        # Cropping to 200 x 300
        return self._random_crop(x, y, (200, 300))

class PCCls(csvDataset):
    n_category = 22
    input_shape = None
    def __init__(self, partition, config) -> None:
        super().__init__(partition, config)

    def _normalize(self, pc):
        """
        np array
        """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    def _sample_pc_from_mesh(self, mesh, num_samples):
        return trimesh.sample.sample_surface(mesh, num_samples)[0]
    def get_all_xy_and_preprocess(self):
        X, Y = [], []
        if "_counter" not in self.dict:
            self.dict["_counter"] = 0
        for cls in tqdm(os.listdir(self.path)):
            ext_cls = os.path.join(self.path, cls)
            if not os.path.isdir(ext_cls):
                continue
            if cls not in self.dict:
                self.dict[cls] = self.dict["_counter"]
                self.dict["_counter"] += 1
            train_set = os.path.join(ext_cls, "train")

            for mesh in os.listdir(train_set):
                if mesh[-4:] != ".off":
                    continue
                ext_mesh = os.path.join(train_set, mesh)
                X.append(ext_mesh)
                Y.append(cls)

        return X, Y
    def read_xy(self, x, y):
        # load mesh
        mesh = trimesh.load(file_obj=open(x), file_type="off")
        # sample from mesh
        pc = self._normalize(self._sample_pc_from_mesh(mesh, 1024))
        return torch.tensor(pc, dtype=torch.float32), self.dict[y]

class HuBMAP_HPA(csvDataset):
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

### Level 2: dataset
class VOC2012(ImgSeg):
    data_path = None
    n_category = 22
    input_shape = None
    def __init__(self, partition, config) -> None:
        super().__init__(partition, config)
    def get_all_xy_and_preprocess(self):
        X, Y = [], []
        for y in tqdm(os.listdir(os.path.join(self.path, "SegmentationClass"))):
            
            name, ext = os.path.splitext(y)
            if ext != ".png" or name[-2:] == "_p":
                continue
            ext_y = os.path.join(self.path, "SegmentationClass", y)
            # invoke preprocess
            ext_y = self.preprocess(ext_y)
            ext_x = os.path.join(self.path, "JPEGImages", name + ".jpg")
            X.append(ext_x)
            Y.append(ext_y)
        return X, Y
    def preprocess(self, seg_path):
        # change occurences of 255 to 21
        y_np = read_img(seg_path)
        y_np[y_np == 255] = 21
        save_path = seg_path + "_p" + ".png"
        write_img(y_np, save_path)
        return save_path
    def read_xy(self, x, y):
        return self.aug_crop(read_img(x) / 255, read_img(y).type(torch.long))


if __name__ == "__main__":
    # for testing
    config = {"data":"VOC2012", "train_val_test_ratio":[0.8, 0.1, 0.1], "exp_name":"test"}
    seg = ImgSeg("train", config)
    x, y = seg[42]
    for i in range(y.shape[1]):
        for j in range(y.shape[2]):
            if y[0, i, j] == 44:
                x[:, i, j] = 0
    plt.imshow(x.permute(1, 2, 0))
    plt.show()
    print(x.shape)
    print(y.shape)
