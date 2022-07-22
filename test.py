from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
def rle2mask(rle: str, height: int, width: int) -> np.ndarray:
    # fd = pd.read_csv("dataset/organ/train.csv")
    # rle = fd["rle"][0]
    # height = fd["img_height"][0]
    # width = fd["img_width"][0]
    mask = np.zeros((height * width,))
    rle_l = rle.strip().split(' ')
    for i in range(0, len(rle_l) - 1, 2):
        # print(i, "/", len(rle_l) - 1)
        start, step = int(rle_l[i]), int(rle_l[i + 1])
        # print(start, step)
        mask[start: start + step] = 1
        # for j in range(start, start + step):
        #     x = j // height
        #     y = j % height
        #     mask[y][x] = 1
    mask = mask.reshape(width, height).T
    plt.imshow(mask)
    # plt.show()
    return mask
def mask2rle(mask: np.ndarray):
    pass
if __name__ == "__main__":
    # label_path = "dataset/organ/train_labels"
    # os.mkdir(label_path)
    # fd = pd.read_csv("dataset/organ/train.csv")
    # for i in range(fd.shape[0]):
    #     rle = fd["rle"][i]
    #     height = fd["img_height"][i]
    #     width = fd["img_width"][i]
    #     id = fd["id"][i]
    #     mask = rle2mask(rle, height, width)
    #     np.savetxt(os.path.join(label_path, "%d.npy" % id), mask)
    train_path = "dataset/organ/train_images"
    annotation_path = "dataset/organ/train_annotations"
    large_images = []
    for train_img in os.listdir(train_path):

        f = open(os.path.join(annotation_path, os.path.splitext(train_img)[0] + ".json"))
        ann_list = json.load(f)
        f.close()
        img = Image.open(os.path.join(train_path, train_img))
        pix = np.array(img) # 0th channel is read 1st channel is green 2nd channel is blue
        print(pix.shape)
        for cycle in ann_list:
            for x,y in cycle:
                pix[y-5:y+5,x-5:x+5, :] = [255, 0, 0]
        plt.imshow(pix)
        plt.show()
        raise Exception("break")
