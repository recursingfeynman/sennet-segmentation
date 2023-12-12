from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets.sennet import InferenceDataset
from ..functional import combine_patches, extract_patches


@torch.no_grad()
def predict(
    model: nn.Module,
    paths: list[str],
    device: str | torch.device,
    config: Any,
) -> np.ndarray:
    """
    Predict segmentation masks.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    paths : list of str
        Paths to source images (D,).
    device : str or torch.device
        Device on which to perform predictions.
    config : any
        The configuration class. Must have transforms, dim, stride, padding, batch_size,
        thresholds and lomc attributes.

    Returns
    -------
    np.array
        Predicted volume [D, H, W].
    """
    transforms = config.transforms
    dim = config.dim
    stride = config.stride
    padding = config.padding
    bs = config.batch_size
    thresholds = config.thresholds
    lomc = config.lomc

    model.eval()
    nthreads = torch.get_num_threads() * 2
    volume = []
    dataset = InferenceDataset(paths, transforms)
    loader = DataLoader(dataset, batch_size=bs, num_workers=nthreads)
    for images in tqdm(loader, desc="Processing"):
        B, C, H, W = images.shape
        patches = extract_patches(images, dim, stride, padding)
        patches = patches.reshape(-1, C, dim, dim)

        with torch.autocast(device_type=str(device)):
            outputs = model.forward(patches.to(device))

        outputs = outputs.sigmoid().cpu()
        outputs = outputs.contiguous().view(B, -1, outputs.size(1), dim, dim)
        outputs = (outputs[:, :, 0:1] * outputs[:, :, 1:2]) > thresholds[0] # V * K

        # Reconstruct original images
        outputs = combine_patches((H, W), outputs.byte(), dim, stride, lomc)
        volume.extend(outputs.squeeze(1).numpy())

    return np.stack(volume)
