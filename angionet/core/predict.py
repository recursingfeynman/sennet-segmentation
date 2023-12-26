from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..functional import combine_patches, extract_patches, rescale
from ..postprocessing import Combine


@torch.no_grad()
def predict(
    model: nn.Module,
    dataset: Dataset,
    dim: int,
    stride: int,
    padding: str,
    batch_size: int,
    tta: Optional[Combine] = None,
    device: str | torch.device = "cpu",
    kidney_model: Optional[nn.Module] = None,
) -> np.ndarray:
    """
    Predict segmentation masks.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    dataset : Dataset
        Inference dataset.
    dim : int
        Window size.
    stride : int
        Stride value for patch extraction.
    padding : str
        Padding mode.
    batch_size : int
        Batch size.
    tta : callable, optional
        Test time augmentations class.
    device : str or torch.device
        Device on which to perform predictions.
    kidney_model : nn.Module, optional
        Model to predict kidney.

    Returns
    -------
    np.array
        Predicted probabilities [D, H, W].
    """
    model.eval()
    volume = []
    nthreads = torch.get_num_threads() * 2
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=nthreads)
    for images in tqdm(loader, desc="Processing"):
        B, C, H, W = images.shape
        patches = extract_patches(images, dim, stride, padding)
        patches = patches.reshape(-1, C, dim, dim)

        if tta is None:
            with torch.autocast(device_type=str(device)):
                output = model.forward(patches.to(device))
            output = output.sigmoid().cpu()
        else:
            output = []
            for group in tta.augment(images): # (T, B * N, C, H, W)
                with torch.autocast(device_type=str(device)):
                    out = model.forward(group.to(device))
                output.append(out.sigmoid().cpu())
            output = tta.disaugment(torch.stack(output)).mean(dim = 0)

        output = output.contiguous().view(B, -1, 1, dim, dim)

        if kidney_model is not None:
            kidneys = find_kidney(kidney_model, images, 512, device)
            kidneys = extract_patches(kidneys, dim, stride, padding="constant")
            output = output * kidneys

        # Reconstruct original images
        outputs = combine_patches((H, W), output.float(), dim, stride, lomc=False)
        volume.extend(outputs.squeeze(1).numpy())

    return np.stack(volume)


def find_kidney(model, images, dim, device):
    model.eval()

    H, W = images.shape[-2:]
    images = torch.stack([rescale(img) for img in images])
    images = F.resize(images, (dim, dim), antialias=False)

    with torch.autocast(device_type=str(device)):
        kidneys = model.forward(images.to(device))

    return F.resize(kidneys.sigmoid().cpu(), (H, W))
