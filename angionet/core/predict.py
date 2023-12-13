from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..functional import combine_patches, extract_patches


@torch.no_grad()
def predict(
    model: nn.Module,
    dataset: Dataset,
    device: str | torch.device,
    config: Any,
) -> np.ndarray:
    """
    Predict segmentation masks.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    dataset : Dataset
        Inference dataset.
    device : str or torch.device
        Device on which to perform predictions.
    config : any
        The configuration class. Must have dim, stride, padding, batch_size,
        threshold, prod and lomc attributes.

    Returns
    -------
    np.array
        Predicted volume [D, H, W].
    """
    dim = config.dim
    stride = config.stride
    padding = config.padding
    bs = config.batch_size
    threshold = config.threshold
    prod = config.prod
    lomc = config.lomc

    model.eval()
    nthreads = torch.get_num_threads() * 2
    volume = []
    loader = DataLoader(dataset, batch_size=bs, num_workers=nthreads)
    for images in tqdm(loader, desc="Processing"):
        B, C, H, W = images.shape
        patches = extract_patches(images, dim, stride, padding)
        patches = patches.reshape(-1, C, dim, dim)

        with torch.autocast(device_type=str(device)):
            outputs = model.forward(patches.to(device))

        outputs = outputs.sigmoid().cpu()
        outputs = outputs.contiguous().view(B, -1, outputs.size(1), dim, dim)

        if prod: # Vessels * Kidney
            outputs = (outputs[:, :, 0:1] * outputs[:, :, 1:2]) > threshold
        else:
            outputs = (outputs[:, :, 0:1] > threshold)

        # Reconstruct original images
        outputs = combine_patches((H, W), outputs.byte(), dim, stride, lomc)
        volume.extend(outputs.squeeze(1).numpy())

    return np.stack(volume)
