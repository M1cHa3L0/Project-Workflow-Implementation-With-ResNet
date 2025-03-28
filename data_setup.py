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

NUM_WORKERS = os.cpu_count()

# get data
def custom_data(data_path, img_path, data_file):
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    image_path = data_path/img_path

    # download data
    with open(data_path/'file', 'wb') as f:
        request = requests.get(data_file)
        print('downloading')
        f.write(request.content)

    # unzip file
    with zipfile.ZipFile(data_path/'file', 'r') as zip_ref:
        print(f'unziping file')
        zip_ref.extractall(image_path)
    
    print('downloaded')
    return image_path/'train', image_path/'test'
    

    

# create dataloader
def create_dataLoader(train_dir, test_dir, transforms, batch_size, num_workers=NUM_WORKERS):
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