"""
utils funciton
save model, load model,
"""
from pathlib import Path
import torch

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
