from importlib import reload
from itertools import product

import numpy as np
import rasterio as rio
import torch
from tqdm import tqdm

from config import config_imp as conf
from scripts import scripts


def add_to_mask(mask, predictions, batch_pos):
    """from the predictions list. it will 'paste' or add every patch into the mask and
    will return the coresponding value"""
    for i in range(len(batch_pos)):

        col, row, width, height = batch_pos[i]

        # here we could create a rule to replace the overlapped values
        # from the new_prediction. instead of replacing the previous value.
        current_value = mask[row : row + height, col : col + width]
        new_prediction = predictions[i].squeeze()

        # this will be used to crop the borders to its original size.
        # we modify the border chips to fit into the model and here we make it back
        new_prediction = new_prediction[:height, :width]
        mask[row : row + height, col : col + width] = new_prediction

    return mask


def detect(model, img, height, width, stride, batch_size, normalize=True):
    """
    Use the model to predict the entire image. It will cut the big window image into
    smaller chips of height*width every stride pixels, then will create a batch that will
    be passed to the model to predict. At the end all the small prediction patches will
    build an individual image.

    """

    nols, nrows = img.meta["width"], img.meta["height"]
    meta = img.meta.copy()
    if "float" not in meta["dtype"]:
        # The prediction is a float so we keep it as float
        # to be consistent with the prediction.
        meta["dtype"] = np.float32

    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = rio.windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    mask = torch.zeros((nrows, nols))

    batch = []
    batch_pos = []

    for col_off, row_off in offsets:

        window = rio.windows.Window(
            col_off=col_off, row_off=row_off, width=width, height=height
        ).intersection(big_window)

        transform = rio.windows.transform(window, img.transform)
        # Add zero padding in case of corner images
        patch = torch.zeros((3, height, width))
        img_sm = img.read(window=window)

        temp_im = np.squeeze(img_sm)
        temp_im = temp_im[[0, 1, 2], ...]
        temp_im = torch.from_numpy(temp_im).to("cuda")

        if normalize:
            temp_im = scripts.normalize_tensor(temp_im)

        patch[..., : window.height, : window.width] = temp_im

        batch.append(patch)
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))

        if len(batch) == batch_size:

            torch_batch = torch.stack(batch).to("cuda")
            outputs = model(torch_batch)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.expand(224, 224, 1, batch_size).permute(
                3, 2, 1, 0
            )

            mask = add_to_mask(mask, predictions, batch_pos)

            batch = []
            batch_pos = []

    # To handle the edge of images as the image size may not be divisible by n complete
    # batches and few frames on the edge may be left.
    if batch:

        mask = add_to_mask(mask, predictions, batch_pos)
        batch = []
        batch_pos = []

    return mask


def save_mask(mask, profile, output_file_name):
    """saves prediction mask in output_file path using the profile."""
    mask = mask.detach().numpy()
    profile["count"] = 1
    profile["dtype"] = np.float32

    with rio.open(output_file_name, "w", **profile) as dst:
        dst.write(mask.astype(profile["dtype"]), 1)
