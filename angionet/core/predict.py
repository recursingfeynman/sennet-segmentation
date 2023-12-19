from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..functional import combine_patches, extract_patches, rescale


@torch.no_grad()
def predict(
    model: nn.Module,
    dataset: Dataset,
    device: str | torch.device,
    config: Any,
    kidney_model: Optional[nn.Module],
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
        threshold and lomc attributes.
    kidney_model : nn.Module, optional
        Model to predict kidney.

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
    lomc = config.lomc
    tta = config.tta

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
        if tta is not None:
            outputs = torch.cat((outputs[None], tta.predict(patches.to(device))), 0)
            outputs = torch.mean(outputs, 0)

        outputs = outputs.contiguous().view(B, -1, 1, dim, dim)

        if kidney_model is not None:
            kidneys = find_kidney(kidney_model, images, 512, device)
            kidneys = extract_patches(kidneys, dim, stride, padding="constant")
            outputs = (outputs * kidneys) > threshold
        else:
            outputs = outputs > threshold

        # Reconstruct original images
        outputs = combine_patches((H, W), outputs.byte(), dim, stride, lomc)
        volume.extend(outputs.squeeze(1).numpy())

    return np.stack(volume)


def find_kidney(model, images, dim, device):
    model.eval()

    H, W = images.shape[-2:]
    images = torch.stack([rescale(img) for img in images])
    images = F.resize(images, (dim, dim))

    with torch.autocast(device_type=str(device)):
        kidneys = model.forward(images.to(device))

    return F.resize(kidneys.sigmoid().cpu(), (H, W))
