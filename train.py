"""
function for train and evaluate model
"""
from typing import Tuple, List, Dict
import torch
import torch.types
from tqdm.auto import tqdm

def train_step(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device) -> Tuple[float,float]:
    """
    trains a pytorch model for a single epoch

    train mode -> forward -> loss -> backward -> gd

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy).
    """
    model.train()
    train_loss, train_acc = 0,0
    for batch, (X, y) in enumerate(train_dataloader):
        # data to target device
        X, y = X.to(device), y.to(device)

        # forward
        train_logits = model(X)

        # loss and acc
        loss = loss_fn(train_logits, y)
        train_loss += loss.item()
        train_label = train_logits.argmax(dim=1)
        train_acc += (train_label == y).sum().item()/len(y)

        # zero grad
        optimizer.zero_grad()
        # backward
        loss.backward()
        # gd
        optimizer.step()
    
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    return train_loss, train_acc


def test_step(
        model:torch.nn.Module,
        test_dataloader:torch.utils.data.DataLoader,
        loss_fn:torch.nn.Module,
        device:torch.device) -> Tuple[float, float]:
    
    """
    test a pytorch model for a single epoch

    eval mode -> forward -> loss

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy).
    """
    # turn off gradient tracking
    model.eval()
    test_loss, test_acc = 0,0
    with torch.inference_mode():        
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)

            test_logits = model(X)
            # test loss
            
            loss = loss_fn(test_logits, y)
            test_loss += loss.item()

            test_label = test_logits.argmax(dim=1)
            # test acc
            test_acc += (test_label==y).sum().item()/len(y)
        
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    return test_loss, test_acc


def train(
        model:torch.nn.Module,
        train_dataloader:torch.utils.data.DataLoader,
        test_dataloader:torch.utils.data.DataLoader,
        loss_fn:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        epochs:int,
        device:torch.device) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    """

    # result dictionary
    result = {
        'train_loss':[],
        'train_acc':[],
        'test_loss':[],
        'test_acc':[]}
    
    for epoch in tqdm(range(epochs)):
        # train
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)

        # test
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        # print result
        print(f'Epoch [{epoch+1}/{epochs}] | train loss: {train_loss:.4f}, train acc: {train_acc*100:.2f}% | test loss: {test_loss:.4f}, test acc: {test_acc*100:.2f}%')
        
        # save result
        result['train_loss'].append(train_loss)
        result['train_acc'].append(train_acc)
        result['test_loss'].append(test_loss)
        result['test_acc'].append(test_acc)
    
    return result


