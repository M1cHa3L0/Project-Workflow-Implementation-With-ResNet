"""
load data and preprocess
create dataloader
visualize data
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import requests
import zipfile
import shutil
from sklearn.model_selection import train_test_split

NUM_WORKERS = os.cpu_count()
TRAIN_RATIO = 0.8
    
# create dataloader
def create_dataLoader(transforms, batch_size, train_dir=None, test_dir=None, train_data=None, test_data=None, num_workers=NUM_WORKERS):
    """
    create train and test dataloader

    Arg:
        train_dir: train data directory
        test_dir: test data directory
        transforms: torchvision transform to perform train and test data
        batch_size:number of sample per batch in DataLoader.
        num_workers: number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)
        class_names is a list of target classes
    """
    if train_dir is not None and test_dir is not None:
        # load data
        train_data = datasets.ImageFolder(
            root=train_dir,
            transform=transforms
            )
    
        test_data = datasets.ImageFolder(
            root=test_dir,
            transform=transforms
        )

    class_names = train_data.classes

    # turn dataset to dataloader
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_dataloader, test_dataloader, class_names

def vizData():
    return 0



# split data file
def split(dataset_dir, output_dir, train_ratio=TRAIN_RATIO):
    """
    turn dataset structure from

    dataset[ class1[ img1.jpg, img2.jpg...], class2[img1.jpg, img2.jpg...],...]
    
    to

    output[ train[ class1[ img1.jpg,...], class2[img1.jpg,...],...], test[ class1[ img1.jpg,...], class2[img1.jpg,...]]]

    Arg:
        dataset_dir: The directory to which you want to change
        output_dir: The directory to which you want to save the data
        train_ratio: Train dataset split rate
    """
    print("Data spliting...")

    # 创建训练和测试集目录
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 遍历每个类别文件夹
    for label in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label)
        
        # 非文件夹
        if not os.path.isdir(label_path):
            continue

        # 获取类别下的所有image
        images = [f for f in os.listdir(label_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # 划分数据集
        train_images, test_images = train_test_split(images, train_size=train_ratio)

        # 复制image到目标目录
        for img in train_images:
            image_path = os.path.join(label_path, img) # image path in dataset
            class_path = os.path.join(train_dir, label) # class path in train data file
            os.makedirs(class_path, exist_ok=True) # 创建类别目录
            shutil.copy(image_path, os.path.join(class_path, img)) # 复制dataset的img到类别目录中

        for img in test_images:
            image_path = os.path.join(label_path, img)
            class_path = os.path.join(test_dir, label)
            os.makedirs(class_path, exist_ok=True)
            shutil.copy(image_path, os.path.join(class_path, img))

    print("Done!")
