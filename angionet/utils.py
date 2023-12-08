import gc
import os
import random
from collections import defaultdict
from typing import Any, Optional

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import clear_output # noqa
from torch.utils.data import DataLoader

from .functional import encode


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Parameters:
    ----------
    seed : int, default=42
        The seed value to set.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def cleanup():
    """
    Perform cleanup operations to free up GPU memory and garbage collect.
    """
    torch.cuda.empty_cache()
    gc.collect()


def prettify_transforms(transforms: dict[str, A.BaseCompose]) -> dict[str, list[str]]:
    """
    Prettify transformations by organizing them into a more readable format.

    Parameters
    ----------
    transforms : dict[str, A.Compose]
        A dictionary representing transformations, where keys are stage
        and values are A.Compose object.

    Returns
    -------
    dict[str, list[str]]
        A prettified dictionary where keys are stage name (train/test) and values 
        are lists of transforms class names.
    """
    ptransforms = defaultdict(list)
    for key, value in transforms.items():
        ptransforms[key].extend([v.get_class_fullname() for v in value])

    return dict(ptransforms)


@torch.no_grad()
def visualize(
    model: nn.Module,
    loader: DataLoader,
    k: int = 8,
    threshold: Optional[float] = None,
    device: str | torch.device = "cpu",
):
    """
    Visualize the model predictions alongside the original images and masks 
    from a DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model instance.
    loader : torch.utils.data.DataLoader
        PyTorch DataLoader instance.
    k : int, default=8
        Number of samples to visualize
    threshold : float, optional
        Compare model outputs with given threshold.
    device : str or torch.device, default='cpu'
        Device to use for inference (e.g., "cpu" or "cuda").
    """
    clear_output(wait=True) # noqa

    images, masks, _ = next(iter(loader))
    indices = np.random.choice(len(images), size=k, replace=False)
    images, masks = images[indices], masks[indices]
    model.eval().to(device)

    with torch.autocast(device_type = str(device)):
        predicted = model.forward(images.to(device))
        
    predicted = predicted.sigmoid()

    if threshold is not None:
        predicted = (predicted > threshold).byte()

    groups = (images, masks[:, 0], predicted[:, 0], masks[:, 1], predicted[:, 1])
    titles = ("Original", "Vessels", "Predicted Vessels", "Kidney", "Predicted Kidney")

    for gr, label in zip(groups, titles):
        _, axs = plt.subplots(1, k, figsize=(15, 6))
        plt.subplots_adjust(wspace=0)
        plt.suptitle(label, y=0.65)
        for idx, ax in enumerate(axs.flatten()):
            ax.imshow(gr[idx].squeeze(0).cpu(), cmap="Greys_r")
            ax.axis("off")
        plt.show()

    cleanup()


def load_volume(dataset: Any) -> dict[str, np.ndarray]:
    """
    Load separate images from dataset as single volume.

    Parameters
    ----------
    dataset : VolumeDataset
        VolumeDataset instance.

    Returns
    -------
    dict[str, np.array]
        Volume.
    """
    D, H, W = len(dataset), dataset.height, dataset.width
    volume = {
        "image": np.zeros((D, H, W), dtype="uint8"),
        "vessels": np.zeros((D, H, W), dtype="uint8"),
        "kidney": np.zeros((D, H, W), dtype="uint8"),
    }

    for d in range(len(dataset)):
        image, vessels, kidney = dataset[d]
        volume["image"][d] = image
        volume["vessels"][d] = vessels
        volume["kidney"][d] = kidney

    return volume


def save_volume(
    group: str, volume: dict[str, np.ndarray], axis: tuple[int, ...]
) -> pd.DataFrame:
    """
    Save loaded volume.

    Parameters
    ----------
    group : str
        Images group. Destination directory with this name will be created.
    volume : dict[str, np.array]
        Volume to save.
    axis : tuple[int, int, int]
        Transpose volume to a given axis.
    
    Returns
    -------
    pd.DataFrame
        Paths and run-length encodings of saved volume.
    """
    os.makedirs(group, exist_ok=True)

    image, vessels, kidney = volume.values()
    image = image.transpose(*axis)
    vessels = vessels.transpose(*axis)
    kidney = kidney.transpose(*axis)

    data: dict[str, list] = {
        "path": [],
        "vessels": [],
        "kidney": [],
    }
    for index in range(len(image)):
        path = f"{group}/{str(index).zfill(4)}.tif"
        cv2.imwrite(path, image[index])

        data["path"].append(path)
        data["vessels"].append(encode(vessels[index]))
        data["kidney"].append(encode(kidney[index]))

    return pd.DataFrame.from_dict(data)
