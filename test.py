"""
main file
1. load data
2. model
3. train
4. test and eval
5. save
"""
import torch
from torch import nn
import os
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor
import data_setup, train, ResNet, utils

# device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# hyperparameters
EPOCHS = 5
BATCH_SIZE = 32
NUM_WORKERS = 0
LR = 0.01
HIDDEN_UNITS = 0
data_file = "https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip"

# data transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    # transforms.TrivialAugmentWide(num_magnitude_bins=31) # data augmentation
])

# load data
train_dir, test_dir = data_setup.custom_data('resnet/data', 'pizza_steak_sushi', data_file)
'''
train_data = datasets.Food101(
    root='resnet/data/food101',
    split='train',
    transform=transform,
    target_transform=False,
    download=True
)

test_data = datasets.Food101(
    root='resnet/data/food101',
    split='test',
    transform=transform,
    download=True)'
    '''
train_dataloader, test_dataloader, class_names = data_setup.create_dataLoader(train_dir, test_dir, transform, BATCH_SIZE, NUM_WORKERS)

print(class_names)



# load model
resnet50model = ResNet.ResNet50(num_classes=len(class_names)).to(device)

# loss func and optimier
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=resnet50model.parameters(), lr=LR)

results = train.train(resnet50model, train_dataloader, test_dataloader, loss_fn, optimizer, EPOCHS, device)
print(results)