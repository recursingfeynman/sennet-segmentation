import gc
import os
import random
from collections import defaultdict
from typing import Any

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import clear_output  # noqa
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from .functional import colorize, encode


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
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
    class_index: int = 0,
    nrow: int = 8,
    figsize: tuple[int, int] = (20, 6),
    return_masked: bool = False,
) -> list[np.ndarray] | None:
    clear_output(wait=True)
    batch = next(iter(loader))
    model.eval()
    with torch.autocast(device_type=str(device)):
        preds = model.forward(batch[0].to(device))
    preds = (preds.sigmoid() > threshold).cpu()

    masked = []
    for index in range(batch[0].shape[0]):
        image = batch[0][index].squeeze().numpy()
        mask = batch[1][index][class_index].numpy().astype("uint8")
        pred = preds[index][class_index].numpy().astype("uint8")
        colorized = torch.from_numpy(colorize(image, mask, pred))
        masked.append(colorized.permute(2, 0, 1))

    grid = make_grid(masked, nrow=nrow)
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

    cleanup()

    if return_masked:
        return [m.permute(1, 2, 0).numpy() for m in masked]
    return None


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
    axis : tuple of ints
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


def unbind(input: np.ndarray, axis: int = 0) -> list[np.ndarray]:
    """
    Removes a numpy array dimension. Same as torch.unbind.

    Parameters
    ----------
    input : np.array
        The array to unbind.
    axis : int
        Dimension to remove.

    Returns
    -------
    np.array
        Returns a tuple of all slices along a given dimension, already without it.

    References
    ----------
    [1] https://stackoverflow.com/questions/64097426
    [2] https://pytorch.org/docs/stable/generated/torch.unbind.html
    """
    return list(np.moveaxis(input, axis, 0))
