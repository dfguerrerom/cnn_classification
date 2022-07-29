import os
from typing import Dict, Tuple

if "GDAL_DATA" in list(os.environ.keys()): del os.environ["GDAL_DATA"]
if "PROJ_LIB" in list(os.environ.keys()): del os.environ["PROJ_LIB"]

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import rasterio as rio
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from torchgeo.datasets import VisionDataset

from . import scripts

device = "cuda" if torch.cuda.is_available() else "cpu"


class PlanetDataSet(VisionDataset):
    
    def __init__(self, 
                 root, 
                 data_df, 
                 label_col, 
                 ext=".tif", 
                 transforms=None, 
                 multilabel=False,
                 fixed_tags=None,
                 sep=","
        ):
        """Initialize a new VisionDataset instance.
        Args:
            root (str): path where the images are located
            data_df (pd.DataFrame): pandas dataframe containing id with image names and 
                labels cols
            label_col (str): name of the column to use as labels column
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            fixed_tags (dict(name:number))
        """
        
        self.transforms = transforms
        self.root = root
        self.data_df = data_df.reset_index(drop=True)
        self.label_col = label_col
        self.ext = ext
        self.multilabel = multilabel
        self.sep = sep
        
        self.classes, self.class_to_idx = self.get_tags()
        
        # Override if we are using some fixed tags
        self.class_to_idx = fixed_tags or self.class_to_idx
        
        self.classes = self.classes["tag"].values

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        image = self._load_image(index)
                        
        if self.multilabel:
            
            label = torch.LongTensor([
                 self.class_to_idx[lbl] for lbl in
                 self.data_df.loc[index][self.label_col].split(self.sep)  
             ])
            label = F.one_hot(label, num_classes=len(self.class_to_idx))
            label = label.sum(dim=0)
            
        else:
            label = torch.tensor(
                [self.class_to_idx[self.data_df.loc[index][self.label_col]]]
            ).type(torch.LongTensor)

        sample = {"image": image.float(), "label": label.to(device)}
        
        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.
        Returns:
            length of the dataset
        """
        return len(self.data_df)
    
    def _load_image(self, index: int) -> Tuple[torch.Tensor]:
        """Load a single image and it's class label.
        Args:
            index: index to return
        Returns:
            the image
            the image class label
        """
        
        image_id = self.data_df.loc[index]["id"]
        image_path = (self.root/image_id).with_suffix(self.ext)
        
        with rio.open(image_path, "r") as src:
            array = src.read()
            tensor = torch.from_numpy(array[[2,1,0],...]).to(device)
            tensor = scripts.normalize_mean_std(tensor)
            return tensor
    
    def plot(self, index, rgb=[2,1,0], display=True):
        """Get image from index and plot it using rgb combination"""
        
        sample = self.__getitem__(index)
        
        # its shape is: Ch.H.W.
        normalized_img = scripts.normalize_tensor(sample["image"])
        # Permute them to be displayed by matplot
        normalized_img = normalized_img.permute(1,2,0)
        # Combine bands in the given rgb order
        normalized_img = normalized_img[..., rgb].cpu().detach()

        if not display: return normalized_img
        
        image_name = self.data_df.loc[index].id
        tag = self.data_df.loc[index][self.label_col]
        
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        
        axes.imshow(normalized_img)
        axes.set_title(f"{tag}\n{image_name}")
        
        return fig.show()
    
    def plot_tags(self, size=20):
        """Plot all tags and images within the dataset"""
        
        tags, _ = self.get_tags()
        tags = tags["tag"].unique()
        
        n_cols = min(len(tags), 4)
        
        fig, axes = plt.subplots(len(tags)//n_cols+1, n_cols, figsize=(size, size))
        
        for idx, tag in enumerate(tags):
            
            image_id = self.data_df[
                self.data_df[self.label_col].isin(tags)
            ].sample().index.values[0]
            
            image_name = self.data_df.loc[image_id].id

            col = idx % n_cols
            row = idx // n_cols
            if isinstance(axes, np.ndarray):
                if isinstance(axes[row], np.ndarray):
                    axes[row][col].set_title(f"{tag}\n{image_name}")
                    axes[row][col].imshow(self.plot(image_id, display=False));
                else:
                    axes[col].set_title(f"{tag}\n{image_name}")
                    axes[col].imshow(self.plot(image_id, display=False));
            else:
                axes.set_title(f"{tag}\n{image_name}")
                axes.imshow(self.plot(image_id, display=False));

        return fig
        
    def get_histogram(self,):
        """"Plot all the classes present in the dataset on an histogram"""
        
        df_tags, _ = self.get_tags()
        fig = px.bar(
            df_tags, 
            x="total", 
            y="tag", 
            orientation="h", 
            color="total",
        )
        
        fig.update_layout(title=f"{self.label_col} distribution")
        
        return fig.show()
        
    def get_tags(self):
        """Get the tags frequency from the original dataset"""
        
        df = self.data_df[self.label_col].copy()
        df = df.astype(str)

        tags = (df
            if not self.multilabel else [
                tag
                for row in df.values
                for tag in row.split(self.sep)
            ]
        )
        
        tags_count = pd.DataFrame(
            list(Counter(tags).items()), columns=["tag", "total"]
        ).sort_values(by="total")
        
        class_to_idx = {
            v:k for k,v in tags_count.reset_index().to_dict()["tag"].items()
        }
        
        return tags_count, class_to_idx
