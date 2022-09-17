# 1. preprocess when creating the dataset
# 2. generate preprocessed data 
# 3. preprocess in __getitem__ x
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
IgnoreFiles = ['.DS_Store']
def split_train_val_test_csv(data_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # try to split at ratio, but if folder explicitly provides test folder, then only train & val is splited
    

    # generate all.csv
    
    # for each folder(class)

    # garbage :( but works somehow
    test_folder = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))][0]
    test_folder = os.path.isdir(os.path.join(test_folder, os.listdir(test_folder)[0]))
    if test_folder:

        train_file = open(os.path.join(data_folder, "all.csv"), "w")
        test_file = open(os.path.join(data_folder, "test.csv"), "w")
        train_file.write("x,y\n")
        test_file.write("x,y\n")

        print("Detected dataset structure: dataset----class 1----test\n\t\t\t\t|\t|\n\t\t\t\t|\t----------train\n\t\t\t\t|\n\t\t\t\t-------class 2")
        for myclass in os.listdir(data_folder):
            ext_myclass = os.path.join(data_folder, myclass)
            if not os.path.isdir(ext_myclass):
                continue
            for exp in os.listdir(ext_myclass):
                extend_exp = os.path.join(ext_myclass, exp)
                if exp in IgnoreFiles:
                    continue
                if (not os.path.isdir(extend_exp)) or (exp not in ["train", "test"]):
                    raise Exception("detected dataset structure not met for: ", os.path.join(ext_myclass, exp))
                if exp == "test":
                    for x in os.listdir(extend_exp):
                        if x in IgnoreFiles:
                            continue
                        test_file.write(os.path.join(myclass, exp, x) + "," + myclass + "\n")
                else:
                    for x in os.listdir(extend_exp):
                        if x in IgnoreFiles:
                            continue
                        train_file.write(os.path.join(myclass, exp, x) + "," + myclass + "\n")
        train_file.close()
        test_file.close()
        all_df = pd.read_csv(os.path.join(data_folder, "all.csv"))
        all_df = all_df.sample(frac=1).reset_index(drop=True)
        
        # if test_ratio is assigned, add it to train_ratio
        train_ratio = train_ratio + test_ratio

        length = all_df.shape[0]
        all_df.iloc[:int(train_ratio * length), :].to_csv(os.path.join(data_folder, "train.csv"), index=False)
        all_df.iloc[int(train_ratio * length):, :].to_csv(os.path.join(data_folder, "val.csv"), index=False)
            
    else:
    # no test folder
        if train_ratio + val_ratio + test_ratio != 1:
            raise Exception("train ratio + val ratio + test ratio should be 1")
        train_file = open(os.path.join(data_folder, "all.csv"), "w")
        train_file.write("x,y\n")
        print("Detected dataset structure: dataset----class 1\n\t\t\t\t|\t\n\t\t\t\t|\t\n\t\t\t\t|\n\t\t\t\t-------class 2")
        for myclass in os.listdir(data_folder):
            ext_myclass = os.path.join(data_folder, myclass)
            if not os.path.isdir(ext_myclass):
                continue
            for x in os.listdir(ext_myclass):
                x = os.path.join(myclass, x)
                train_file.write(x + "," + myclass + "\n")
        train_file.close()
        all_df = pd.read_csv(os.path.join(data_folder, "all.csv"))
        all_df = all_df.sample(frac=1).reset_index(drop=True)
        # print(all_df.info)
        length = all_df.shape[0]
        all_df.iloc[:int(train_ratio * length), :].to_csv(os.path.join(data_folder, "train.csv"), index=False)
        all_df.iloc[int(train_ratio * length) : int((train_ratio + val_ratio) * length), :].to_csv(os.path.join(data_folder, "val.csv"), index=False)
        all_df.iloc[int((train_ratio + val_ratio) * length): , :].to_csv(os.path.join(data_folder, "test.csv"), index=False)
    


def _generate_all_csv(data_folder):
    """
    Generate a csv file for all the images with columns:

    path, label

    Given the original folder is organized by class


    TODO: Make this more "intelligent"

    """
    

class HuBMAP_HPA:
    def __init__(self, data_folder) -> None:
        self.data_folder = data_folder
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
                new_rle = self.mask2rle(mask).strip()
                # # for i in range(len(new_rle)):
                # #     print(new_rle[i], rle[i])
                # # # print(rle)
                # # # print(new_rle)
                assert(rle == new_rle)
                # raise Exception("break")

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
    # def mask2rle(self, mask: np.ndarray) -> str:
    #     """
    #     Convert label format from 2D np.ndarray masks to rle
    #     """
    #     rle = ""
    #     height, width = mask.shape
    #     flat_mask = mask.T.flatten()
    #     i = 0
    #     while i < flat_mask.shape[0]:
    #         start_idx = i
    #         num = 0
    #         while flat_mask[i] == 1:
    #             num += 1
    #         rle += "%d %d" % (start_idx, num)
    #     print(rle)
    #     return rle
    def mask2rle(self, img: np.ndarray) -> str:
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels= img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        print(np.where(pixels[1:] != pixels[:-1]))
        raise Exception("break")
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
    # my_dataset = HuBMAP_HPA("dataset/organ")
    # my_dataset.preprocess()
    # # my_dataset.rle2mask()
    # my_dataset.visualize(1)
    split_train_val_test_csv(os.path.join("/data", "ModelNet40"))
    # split_train_val_test_csv(os.path.join("dataset", "lung"), train_ratio=0.8, val_ratio=0.2, test_ratio=0)