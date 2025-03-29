"""
utils funciton
save model, load model,
"""
from pathlib import Path
import torch
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd

def save_model(model:torch.nn.Module,
               dir:str,
               model_name:str):
    """
    save model to target directory

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
    """

    # create directory
    path_name = Path(dir)
    path_name.mkdir(parents=True, exist_ok=True)

    # create model path
    assert model_name.endswith('.pth') or model_name.endswith('.pt'), 'model name must ends with .pth or .pt'
    model_path = path_name / model_name

    # save model
    print(f'save model to {model_path}')
    torch.save(obj=model.state_dict(), f=model_path)


def plot_loss_curve(results: Dict[str, List[float]]):
    results = pd.DataFrame(results)
    train_loss = results['train_loss']
    train_acc = results['train_acc']
    test_loss = results['test_loss']
    test_acc = results['test_acc']
    plt.figure(figsize=(15,5))
    epoch = range(len(results))
    plt.subplot(1,2,1)
    plt.plot(epoch, train_loss, label='train loss')
    plt.plot(epoch, test_loss, label='test loss')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epoch, train_acc, label='train accuracy')
    plt.plot(epoch, test_acc, label='test accuracy')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.legend()
