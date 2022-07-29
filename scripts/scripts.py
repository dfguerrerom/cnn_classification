from pathlib import Path

import torch

from config import conf
from config.settings import settings


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

def get_model_path(TEST_NAME, model_id, timestr):
    """Creates a model path based on the model settings."""
    
    setting = settings[TEST_NAME]
    
    model_name = setting.model.name
    optimizer_name = setting.optimizer.name
    loss_fn_name = setting.loss_fn.name
    transfer_l = None
    
    if setting.model.transfer:
        # if there is fixed_Feature will return string fixed feature, else will fallback to fine_tune
        transfer_l = (setting.model.get("fixed_feature") and "fixed_feature") or "fine_tune"
    
    path = Path(
        conf.out_model_path/f"{model_name}_{model_id}_{timestr}_{optimizer_name}_{loss_fn_name}_{transfer_l}"
    )
    
    return path