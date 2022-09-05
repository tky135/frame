import struct
from cv2 import split
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import metrics
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def readMNIST(file):
    f = open(file, "rb")
    magic, num = struct.unpack(">ii", f.read(8))

    if magic == 2049:
        # lable file
        y = np.zeros(num, dtype=np.int32)
        for i in range(num):
            y[i] = struct.unpack("b", f.read(1))[0]
        return y
    elif magic == 2051:
        row, col = struct.unpack(">ii", f.read(8))
        x = np.zeros((num, row, col), dtype=np.int32)
        for k in range(num):
            for j in range(col):
                for i in range(row):
                    x[k, j, i] = struct.unpack("B", f.read(1))[0]
        return x
    else:
        return None
    
    f.close()

def clean_cifar():
    path = os.path.join("/data", "cifar-10")
    f = open(os.path.join(path, "trainLabels.csv"))
    g = open(os.path.join(path, "train.csv"), 'w')
    f.readline()
    g.write("image,label\n")
    for line in f.readlines():
        idd, label = line.split(',')
        g.write("train/" + idd + ".png," + label)

    f.close()
    g.close()
    h = open(os.path.join(path, "test.csv"), 'w')
    h.write("image\n")
    for img in os.listdir(os.path.join(path, "test")):
        h.write("test/" + img + '\n')
    h.close()
        
def plot_confusion_matrix(confusion_matrix, labels):
    display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = labels)
    display.plot()
    plt.savefig("confusion_matrix.jpg")
if __name__ == "__main__":
    plot_confusion_matrix(np.array([[0, 1], [2, 3]]), ["yes", "no"])