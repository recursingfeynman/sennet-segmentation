from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets.sennet import InferenceDataset
from ..functional import combine_patches, encode, extract_patches
from ..postprocessing import postprocess


@torch.no_grad()
def predict(
    model: nn.Module,
    frame: pd.DataFrame,
    transforms: A.BaseCompose,
    device: str | torch.device,
    config: Any,
):
    """
    Predict segmentation masks.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    frame : pd.DataFrame
        DataFrame containing paths to input images.
    transforms : A.BaseCompose
        Image transforms.
    device : str or torch.device
        Device on which to perform predictions.
    config : Any
        Configuration class. Should have dim, stride, padding, batch_size and thresholds
        attributes.

    Returns
    -------
    list[str]
        List of run-length encoded masks.
    """
    dim = config.dim
    stride = config.stride
    padding = config.padding
    bs = config.batch_size
    thresholds = config.thresholds

    model.eval()
    encodings = []
    nthreads = torch.get_num_threads() * 2
    for name, group in frame.groupby("group"):
        volume = []
        dataset = InferenceDataset(group.path.values, transforms)
        loader = DataLoader(dataset, batch_size=bs, num_workers=nthreads)
        for images in tqdm(loader, desc=f"Processing {name}"):
            B, C, H, W = images.shape
            patches = extract_patches(images, dim, stride, padding)
            patches = patches.reshape(-1, C, dim, dim)

            with torch.autocast(device_type=str(device)):
                outputs = model.forward(patches.to(device))

            outputs = outputs.sigmoid().cpu()
            outputs = outputs.contiguous().view(B, -1, outputs.size(1), dim, dim)
            outputs = torch.mul(*outputs.unbind(2)).unsqueeze(2) > thresholds[0]

            volume.extend(
                combine_patches(
                    shape=(H, W),
                    patches=outputs.byte(),
                    dim=dim,
                    stride=stride,
                    lomc=True,
                )
                .squeeze()
                .numpy()
            )

        volume = postprocess(np.stack(volume), threshold=16, connectivity=26)

        for mask in volume:
            encodings.append(encode(np.asarray(mask, dtype=np.uint8)))

    return encodings
