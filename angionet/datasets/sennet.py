from typing import Any, Sequence

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..functional import decode, rescale
from ..utils import unbind


class TrainDataset(Dataset):
    """
    PyTorch dataset for training.

    Parameters
    ----------
    paths : sequence of str
        List of file paths to train images.
    transforms : A.BaseCompose
        Albumentations Compose object.
    normalize : callable, optional
        The normalization function. If not specified, perform min-max normalization
        as default.

    Attributes
    ----------
    paths : sequence of str
        List of file paths to train images.
    transforms : A.BaseCompose
        Albumentations Compose object.
    normalize : callable, optional
        The normalization function. By default perform min-max normalization.
    """

    def __init__(
        self,
        paths: Sequence[str],
        transforms: A.BaseCompose,
        normalize: Any = None,
    ):
        self.paths = paths
        self.transforms = transforms
        self.normalize = normalize if normalize is not None else rescale

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        batch = [
            np.asarray(item, dtype=np.float32)
            for item in np.load(self.paths[index]).values()
        ]

        augs = self.transforms(image=batch[0], masks=[*unbind(np.stack(batch[1:]), 0)])
        image = self.normalize(augs["image"])
        masks = [torch.stack(augs["masks"][:len(batch[1])])]

        if len(augs["masks"]) > 2:
            masks.append(torch.stack(augs["masks"][len(batch[1]):]))

        return image, *masks


class InferenceDataset(Dataset):
    """
    PyTorch dataset for inference.

    Parameters
    ----------
    paths : sequence of str
        List of file paths to test images.
    transforms : A.BaseCompose
        Albumentations Compose with image augmentations.
    normalize : callable, optional
        The normalization function. If not specified, perform min-max normalization
        as default.
    """

    def __init__(
        self, paths: Sequence[str], transforms: A.BaseCompose, normalize: Any = None
    ):
        self.paths = paths
        self.transforms = transforms
        self.normalize = normalize if normalize is not None else rescale

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = cv2.imread(self.paths[index], cv2.IMREAD_GRAYSCALE)
        image = np.asarray(image, np.float32)

        image = self.transforms(image=image)["image"]
        image = self.normalize(image)

        return image


class VolumeDataset(Dataset):
    """
    Initialize VolumeDataset.

    Parameters
    ----------
    frame : pd.DataFrame
        DataFrame containing data.
    group : int
        Group identifier for data filtering.

    Attributes
    ----------
    paths : sequence of str
        Paths to images.
    vessels : sequence of str
        Run-length encoded vessels masks.
    kidney : sequence of str
        Run-length encoded kidney masks.
    height : int
        Height of input images.
    width : int
        Width of input images.
    """

    def __init__(self, frame: pd.DataFrame, group: str):
        frame = frame.loc[frame.group == group]
        self.paths = frame.path.values
        self.vessels = frame.vessels.values
        self.kidney = frame.kidney.values
        self.height = frame.height.iloc[0]
        self.width = frame.width.iloc[0]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[np.ndarray, ...]:
        image = cv2.imread(self.paths[index], cv2.IMREAD_GRAYSCALE)
        image = np.asarray(image, dtype="uint8")

        vessels = decode(self.vessels[index], image.shape)
        kidney = decode(self.kidney[index], image.shape)

        return image, vessels, kidney
