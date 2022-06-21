# 1. preprocess when creating the dataset
# 2. generate preprocessed data 
# 3. preprocess in __getitem__ x
import numpy as np
import pandas as pd
import os
def cvtFloatImg(x):
    return x.astype(np.float32)/255


def getData():
    import torchvision
    mnist_train = torchvision.datasets.FashionMNIST(root="dataset", train=False, download=True)
def procHouse(x, partition="train"):
    y = None
    if partition == "train" or partition == "eval":
        y = x["Sold Price"]
        y = (y - y.mean()) / y.std()
        x = x.drop(["Sold Price", "Id"], axis=1)
    else:
        x = x.drop(["Id"], axis=1)

    x = x.select_dtypes(np.number)
    for col in x.columns:
        x[col].fillna(x[col].mode()[0], inplace=True)
        # if col != ""
        x[col] -= x[col].mean()
        x[col] /= x[col].std()
    mis_val = x.isna().sum().sort_values(ascending=False)
    mis_val = len(mis_val[mis_val != 0].index)
    print("NaN value: ", mis_val)
    return x.values.astype(np.float32), y.values.astype(np.float32)
def pred2l(partition):
    train_data = pd.read_csv("dataset/HousePrice/train.csv")
    test_data = pd.read_csv("dataset/HousePrice/test.csv")

    redundant_cols = ['Address', 'Summary', 'City', 'State', 'Zip']
    test_data.drop(redundant_cols, axis=1, inplace=True)
    train_data.drop(redundant_cols, axis=1, inplace=True)

    large_vel_cols = ['Lot', 'Total interior livable area', 'Tax assessed value', 'Annual tax amount', 'Listed Price', 'Last Sold Price']
    for c in large_vel_cols:
        train_data[c] = np.log(train_data[c]+1)
        test_data[c] = np.log(test_data[c]+1)
    
    y = train_data["Sold Price"]
    train_data.drop(["Sold Price"], axis=1, inplace=True)

    all_features = pd.concat((train_data, test_data), axis=0)
    all_features.drop(["Id"], axis=1, inplace=True)

    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # for in_object in all_features.dtypes[all_features.dtypes=='object'].index:
    #     print(in_object.ljust(20),len(all_features[in_object].unique()))

    features = list(numeric_features)
    features.extend(['Type','Bedrooms'])
    all_features = all_features[features]
    all_features = pd.get_dummies(all_features, dummy_na=True)

    if partition == "train" or partition == "eval":
        return all_features[:train_data.shape[0]].values.astype(np.float32), y.values.astype(np.float32)
    else:
        return all_features[train_data.shape[0]:].values.astype(np.float32)

def reformat_class_folder(data_folder):
    """
    Generate a csv file for all the images with columns:

    path, label

    Given the original folder is organized by class


    TODO: Make this more "intelligent"

    """
    train_file = open(os.path.join(data_folder, "train.csv"), "w")
    train_file.write("image,label\n")
    for myclass in os.listdir(data_folder):
        class_folder = os.path.join(data_folder, myclass)
        if not os.path.isdir(class_folder):
            continue
        for img in os.listdir(class_folder):
            img = os.path.join(myclass, img)
            train_file.write(img + "," + myclass + "\n")


    train_file.close()






if __name__ == "__main__":
    reformat_class_folder(os.path.join("dataset", "lung"))