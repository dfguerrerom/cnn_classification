import time
from pathlib import Path

import torch

from config import conf


def normalize_tensor(x):
    """Min/max normalize to [0, 1] range based on a min-max values for each band."""
    
    # It will expect the band chanel in the first axe, as received from rasterio.
    
    bands = x.shape[0]
    x = x.squeeze()
    
    x_min = torch.min(torch.min(x,dim=(1)).values, dim=1).values.reshape(bands,1,1)
    x_max = torch.max(torch.max(x,dim=(1)).values, dim=1).values.reshape(bands,1,1)
    
    return (x - x_min) / (x_max - x_min)

def normalize_mean_std(x):
    """Min/max normalize to [0, 1] range based on a min-max values for each band."""
    
    # It will expect the band chanel in the first axe, as received from rasterio.
    
    bands = x.shape[0]
    x = x.squeeze()
    
    mean = x.mean(dim=(1,2)).reshape(bands, 1, 1)
    std = x.std(dim=(1,2)).reshape(bands, 1, 1)
    
    return (x - mean) / std

def save_model(model, model_id, optimizer_name, loss_fn_name, suffix):
    """Creates """
    timestr = time.strftime("%Y%m%d-%H%M")
    path = Path(conf.out_model_path/f"Resnet18_{model_id}_{timestr}_{optimizer_name}_{loss_fn_name}_{suffix}")
    torch.save(model.state_dict(), path)
    
    print(f"Saving model...{path.stem}")
