import os
import numpy as np
# from util import readMNIST
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
import pickle
### Level 0: base dataset class
class csvDataset(Dataset):
    fmt = "csv"
    def __init__(self, partition, config) -> None:
        print(super().__init__)
        print(type(self).__mro__)
        print(super(csvDataset, self).__init__)
        super().__init__()
        self.partition = partition
        self.config = config
        print(config)
        # get self.path
        if "data_path" in config["arg_from_data"] and config["arg_from_data"]["data_path"] is not None and os.path.isdir(config["arg_from_data"]["data_path"]):
            self.path = config["arg_from_data"]["data_path"]
        else:
            self.path = os.path.join(config["dataroot"], config["data"].__name__)
            if not os.path.isdir(self.path):
                # warnings.warn("data path %s not found, creating a new one. " % "self.path")
                # os.makedirs(self.path)
                print("ERROR: self.path = ", self.path, "not found. ", file=sys.stderr)
                raise Exception("Path Not Found")
        
        # loading self.dict_store if exists
        self.dict_store = os.path.join(self.path, "dict_store.json")
        self.dict = {}
        if os.path.exists(self.dict_store):
            self.dict = json.load(open(self.dict_store))

        # generate partition.csv if not exist
        if not os.path.exists(os.path.join(self.path, self.partition + ".csv")):
            if hasattr(self, "list_" + self.partition + "_data"):
                xs_list = getattr(self, "list_" + self.partition + "_data")()
                partition_csv = open(os.path.join(self.path, self.partition + ".csv"), "w")
                if type(xs_list) != tuple:
                    xs_list = (xs_list, )
                df = pd.DataFrame(zip(*xs_list))
                df.to_csv(partition_csv, index=False)
                partition_csv.close()

            # if do_split is set to True, or no valid csv file, split the dataset
            else:
                if config["do_split"] == True:
                    print("WARNING: " + os.path.join(self.path, partition + ".csv") + " does not exist and method list_" + self.partition + "_data is not defined, do split?(May overwrite other existing csv)(y/n)", file=sys.stderr)
                    user = input()
                    if user not in ["y", "Y"]:
                        raise Exception("Canceled")
                    self._split_train_val_test_csv()
                    config["do_split"] = False  # only do_split once in one experiment
        
        # read csv file into a dataframe
        
        if self.fmt == "csv":
            df = pd.read_csv(os.path.join(self.path, partition + ".csv"))
            self.xs = df.values
        elif self.fmt == "pkl":
            self.xs = pickle.load(open(os.path.join(self.path, partition + ".pkl"), "rb"))

        # store dictionary to dict_store
        # if not os.path.exists(self.dict_store):
        json.dump(self.dict, open(self.dict_store, "w"))    # user can store information using self.dict
    def __getitem__(self, index):
        # not considering partition for now
        return self.read_data(*(self.xs[index]))
        # when testing or inferencing, now sure what to do


    def __len__(self):
        return len(self.xs)
    def _split_train_val_test_csv(self) -> None:
        """
        This should be a function of class dataset
        """
        train_ratio, val_ratio, test_ratio = self.config["train_val_test_ratio"]
        if train_ratio + val_ratio + test_ratio != 1:
            # raise Exception("train ratio + val ratio + test ratio should be 1")
            pass
        self._generate_all_csv()

        if self.fmt == "csv":
            all_df = pd.read_csv(os.path.join(self.path, "all.csv"))

            all_df = all_df.sample(frac=1).reset_index(drop=True)
            # print(all_df.info)
            length = all_df.shape[0]
            all_df.iloc[:int(train_ratio * length), :].to_csv(os.path.join(self.path, "train.csv"), index=False)
            all_df.iloc[int(train_ratio * length) : int((train_ratio + val_ratio) * length), :].to_csv(os.path.join(self.path, "val.csv"), index=False)
            all_df.iloc[int((train_ratio + val_ratio) * length): , :].to_csv(os.path.join(self.path, "test.csv"), index=False)
        elif self.fmt == "pkl":
            all_list = pickle.load(open(os.path.join(self.path, "all.pkl"), "rb"))
            length = len(all_list)
            # shuffle all_list
            np.random.shuffle(all_list)
            pickle.dump(all_list[:int(train_ratio * length)], open(os.path.join(self.path, "train.pkl"), "wb"))
            pickle.dump(all_list[int(train_ratio * length) : int((train_ratio + val_ratio) * length)], open(os.path.join(self.path, "val.pkl"), "wb"))
            pickle.dump(all_list[int((train_ratio + val_ratio) * length): ], open(os.path.join(self.path, "test.pkl"), "wb"))
        else:
            raise Exception("Unknown format %s" % self.fmt)
    def _generate_all_csv(self):
        if self.fmt == "csv":
            
            allcsv = open(os.path.join(self.path, "all.csv"), "w")
            xs_list = self.list_data() # a tuple of iterables
            if type(xs_list) != tuple:
                xs_list = (xs_list, )
            df = pd.DataFrame(zip(*xs_list))
            df.to_csv(allcsv, index=False)
            allcsv.close()
        elif self.fmt == "pkl":
            xs_list = self.list_data() # a tuple of iterables
            if type(xs_list) != tuple:
                xs_list = (xs_list, )
            
            pickle.dump(list(zip(*xs_list)), open(os.path.join(self.path, "all.pkl"), "wb"))
        else:
            raise Exception("Unknown format %s" % self.fmt)

### Level 1: task-specific dataset
class ImgCls(csvDataset):
    def __init__(self, partition, config) -> None:
        super().__init__(partition, config)

    def list_data(self):
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
    def read_data(self, x, y):
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


    train_metric_list = [class_acc]
    val_metric_list = [class_acc]
    loss_fn = CEloss
    n_inputs = 1

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
    def list_data(self):
        raise Exception("Not Implemented")
    def read_data(self, x, y):
        # load mesh
        mesh = trimesh.load(file_obj=open(x), file_type="off")
        # sample from mesh
        pc = self._normalize(self._sample_pc_from_mesh(mesh, 1024))
        return torch.tensor(pc, dtype=torch.float32), self.dict[y]
class PCSeg(csvDataset):
    data_path = "/data/ShapeNetPart"
    def __init__(self, partition, config) -> None:
        super().__init__(partition, config)
        return
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
    def _sample_pc_with_label(self, pc, label, num_samples):
        """
        pc: np array with shape [n, 3]
        label: np array with shape [n]
        """
        print(pc.shape, label.shape)
        assert pc.shape[0] == label.shape[0]
        n = pc.shape[0]
        indices = np.random.choice(n, num_samples, replace=False)
        return pc[indices], label[indices]
class ShapeNetPart(PCSeg, csvDataset):
    def __init__(self, partition, config) -> None:
        super().__init__(partition, config)

    def list_train_data(self):
        X, C, Y = [], [], []
        if "_counter" not in self.dict:
            self.dict["_counter"] = 0
            self.dict["int2str"] = {}
            self.dict["cat_start"] = {"_counter": 0}
        train_path = os.path.join(self.data_path, "train_data")
        for cat in os.listdir(train_path):
            cat_path = os.path.join(train_path, cat)
            if cat not in self.dict:
                # new category
                self.dict[cat] = self.dict["_counter"]
                self.dict["_counter"] += 1          # make this into a service? Yes!
                self.dict["int2str"][self.dict[cat]] = cat
                self.dict["cat_start"][cat] = self.dict["cat_start"]["_counter"]
                # count the number of parts in a category
                maximum_label = 0
                for label in os.listdir(os.path.join(self.data_path, "train_label", cat))[:10]:
                    label_path = os.path.join(self.data_path, "train_label", cat, label)
                    maximum_label = max(int(np.max(np.loadtxt(label_path))), maximum_label)
                self.dict["cat_start"]["_counter"] += maximum_label
            for pc in os.listdir(cat_path):
                X.append(os.path.join(cat_path, pc))
                C.append(self.dict[cat])
                Y.append(os.path.join(self.data_path, "train_label", cat, os.path.splitext(pc)[0] + ".seg"))
        return X, C, Y
    def list_val_data(self):
        X, C, Y = [], [], []
        if "_counter" not in self.dict:
            self.dict["_counter"] = 0
        val_path = os.path.join(self.data_path, "val_data")
        for cat in os.listdir(val_path):
            cat_path = os.path.join(val_path, cat)
            if cat not in self.dict:
                self.dict[cat] = self.dict["_counter"]
                self.dict["_counter"] += 1          # make this into a service? 
            for pc in os.listdir(cat_path):
                X.append(os.path.join(cat_path, pc))
                C.append(self.dict[cat])
                Y.append(os.path.join(self.data_path, "val_label", cat, os.path.splitext(pc)[0] + ".seg"))
        return X, C, Y
    def list_test_data(self):
        X, C, Y = [], [], []
        if "_counter" not in self.dict:
            self.dict["_counter"] = 0
        test_path = os.path.join(self.data_path, "test_data")
        for cat in os.listdir(test_path):
            cat_path = os.path.join(test_path, cat)
            if cat not in self.dict:
                self.dict[cat] = self.dict["_counter"]
                self.dict["_counter"] += 1          # make this into a service? 
            for pc in os.listdir(cat_path):
                X.append(os.path.join(cat_path, pc))
                C.append(self.dict[cat])
                Y.append(os.path.join(self.data_path, "test_label", cat, os.path.splitext(pc)[0] + ".seg"))
        return X, C, Y

    def read_data(self, x, c, y):

        x = np.loadtxt(x)
        y = np.loadtxt(y)
        x = self._normalize(x)
        x, y = self._sample_pc_with_label(x, y, 1024)
        y = y + self.dict["cat_start"][self.dict["int2str"][str(c)]] - 1
        # make one-hot vector
        c_one_hot = torch.zeros(self.dict["_counter"], dtype=torch.long)
        c_one_hot[c] = 1
        return torch.tensor(x, dtype=torch.float32), c_one_hot, torch.tensor(y, dtype=torch.long)



def test_dataset(class_name):
    import inspect
    from torch.utils.data import DataLoader
    config = {"data": class_name, "train_val_test_ratio": [0.8, 0.2, 0], "dataroot": "/foobar", "do_split": True}
    config["arg_from_data"] = dict([m for m in inspect.getmembers(class_name) if not (inspect.ismethod(m[1]) or m[0].startswith('_'))])     # risky, but let's keep it this way for now
    # __init__
    inst = class_name("train", config)
    # __getitem__
    print(inst[0])
    # data loader
    inst_loader = DataLoader(inst, 32, shuffle=True, drop_last=False, num_workers=8)
    for data in inst_loader:
        for d in data:
            print(d.shape)
        break
class ModelNet10(PCCls):
    n_category = 10
    data_path = "ModelNet10"
    def __init__(self, partition, config) -> None:
        super().__init__(partition, config)
    def list_data(self):
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
    def read_data(self, x, y):
        # load mesh
        mesh = trimesh.load(file_obj=open(x), file_type="off")
        # sample from mesh
        pc = self._normalize(self._sample_pc_from_mesh(mesh, 1024))
        return torch.tensor(pc, dtype=torch.float32), self.dict[y]



class AutoRegress(csvDataset):
    train_metric_list = []
    val_metric_list = []
    loss_fn = neg_log_likelihood
    n_inputs = 1
    n_values = 2
    n_dims = 28 * 28
    fmt = "pkl"
    def __init__(self, partition, config) -> None:
        super().__init__(partition, config)
    def list_data(self):
        # Find all inputs and outputs
        # return a tuple of lists, each list being an input/output
        # what's inside each list is user defined, but the object  should be small. (path to actual object)

        # dataset 1
        # count = 5000
        # rand = np.random.RandomState(0)
        # samples = 0.4 + 0.1 * rand.randn(count)
        # data = np.digitize(samples, np.linspace(0.0, 1.0, 20))

        # dataset 2
        # count = 10000
        # rand = np.random.RandomState(0)
        # a = 0.3 + 0.1 * rand.randn(count)
        # b = 0.8 + 0.05 * rand.randn(count)
        # mask = rand.rand(count) < 0.5
        # samples = np.clip(a * mask + b * (1 - mask), 0.0, 1.0)
        # data = np.digitize(samples, np.linspace(0.0, 1.0, 100))

        # dateset 3
        # from PIL import Image
        # from urllib.request import urlopen
        # import io
        # import itertools

        # im = Image.open("smiley.jpg").resize((self.n_values, self.n_values)).convert('L')
        # im = np.array(im).astype('float32')
        # dist = im / im.sum()

        # pairs = list(itertools.product(range(self.n_values), range(self.n_values)))
        # idxs = np.random.choice(len(pairs), size=10000, replace=True, p=dist.reshape(-1))
        # samples = [pairs[i] for i in idxs]
        # return samples

        # dataset 4
        # from PIL import Image
        # from urllib.request import urlopen
        # import io
        # import itertools

        # im = Image.open("geoffrey=hinton.jpg").resize((self.n_values, self.n_values)).convert('L')
        # im = np.array(im).astype('float32')
        # dist = im / im.sum()

        # pairs = list(itertools.product(range(self.n_values), range(self.n_values)))
        # idxs = np.random.choice(len(pairs), size=100000, replace=True, p=dist.reshape(-1))
        # samples = [pairs[i] for i in idxs]
        # return samples

        # dataset 5
        import pickle
        with open("mnist.pkl", "rb") as f:
            data = pickle.load(f)
        all_data = np.concatenate([data['train'], data['test']], axis=0)
        # Binarize MNIST and shapes dataset
        all_data = (all_data > 127.5).astype('uint8')
        return list(all_data)
    def read_data(self, x):
        # How to read the actual input and output from the lists
        # The inputs to this function is exactly the the outputs of list_data at certain index
        # output is exactly what is given to model at certain index
        ts = torch.tensor(x).flatten().type(torch.long)
        return ts



### Level 2: dataset
class VOC2012(ImgSeg):
    data_path = None
    n_category = 22
    input_shape = None
    def __init__(self, partition, config) -> None:
        super().__init__(partition, config)
    def list_data(self):
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
    def read_data(self, x, y):
        return self.aug_crop(read_img(x) / 255, read_img(y).type(torch.long))

if __name__ == "__main__":
    test_dataset(ShapeNetPart)
    # for testing
    # config = {"data":"VOC2012", "train_val_test_ratio":[0.8, 0.1, 0.1], "exp_name":"test"}
    # seg = ImgSeg("train", config)
    # x, y = seg[42]
    # for i in range(y.shape[1]):
    #     for j in range(y.shape[2]):
    #         if y[0, i, j] == 44:
    #             x[:, i, j] = 0
    # plt.imshow(x.permute(1, 2, 0))
    # plt.show()
    # print(x.shape)
    # print(y.shape)
