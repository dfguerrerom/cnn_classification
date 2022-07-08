from typing import Dict, Tuple

import os

if "GDAL_DATA" in list(os.environ.keys()): del os.environ["GDAL_DATA"]
if "PROJ_LIB" in list(os.environ.keys()): del os.environ["PROJ_LIB"]
from matplotlib import pyplot as plt
from pathlib import Path
import rasterio as rio
import torch
from torchgeo.datasets import VisionDataset
from . import scripts


class PlanetDataSet(VisionDataset):
    
    def __init__(self, root, data_df, label_col, ext=".tif"):
        """Initialize a new VisionClassificationDataset instance.
        Args:
            root (str): path where the images are located
            data_df (pd.DataFrame): pandas dataframe containing id with image names and labels cols
            label_col (str): name of the column to use as labels column
        """
        
        self.root = root
        self.data_df = data_df
        self.label_col = label_col
        self.ext = ext

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        label = self.data_df.iloc[index][self.label_col]
        
        sample = {"image": image, "label": label}
        
        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.
        Returns:
            length of the dataset
        """
        return len(self.data_df)

    def _load_image(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single image and it's class label.
        Args:
            index: index to return
        Returns:
            the image
            the image class label
        """
        
        image_id = self.data_df.iloc[index]["id"]
        image_path = (self.root/image_id).with_suffix(self.ext)
        
        with rio.open(image_path, "r") as src:
            array = src.read()
            tensor = torch.from_numpy(array)
            return tensor
    
    def plot(self, index, rgb=[3,2,1]):
        
        sample = self.__getitem__(index)
        
        # its shape is: Ch.H.W.
        normalized_img = scripts.normalize_tensor(sample["image"])
        
        normalized_img = normalized_img.permute((1,2,0))
        
        # Combine bands in the given rgb order
        normalized_img = normalized_img[..., rgb]
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
        ax.set_title(sample["label"])
        ax.imshow(normalized_img)