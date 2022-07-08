import torch

def normalize_tensor(x):
    """Min/max normalize to [0, 1] range based on a min-max values for each band."""
    
    # It will expect the band chanel in the first axe, as received from rasterio.
    
    bands = x.shape[0]
    x = x.squeeze()
    
    x_min = torch.min(torch.min(x,dim=(1)).values, dim=1).values.reshape(bands,1,1)
    x_max = torch.max(torch.max(x,dim=(1)).values, dim=1).values.reshape(bands,1,1)
    
    return (x - x_min) / (x_max - x_min)