from typing import Sequence

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..functional import decode


class TrainDataset(Dataset):
    """
    PyTorch dataset for training.

    Parameters
    ----------
    paths : Sequence[str]
        List of file paths to train images.
    transforms : A.BaseCompose
        Albumentations Compose with image augmentations.
    """

    def __init__(self, paths: Sequence[str], transforms: A.BaseCompose):
        self.paths = paths
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        image, masks, dtms = [
            np.asarray(item, dtype=np.float32)
            for item in np.load(self.paths[index]).values()
        ]

        augs = self.transforms(
            image=image, masks=[masks[0], masks[1], dtms[0], dtms[1]]
        )

        image = augs["image"]
        masks = torch.stack(augs["masks"][:2])
        dtms = torch.stack(augs["masks"][2:])

        image = (image - image.mean()) / (image.std() + 1e-6)

        return image, masks, dtms


class InferenceDataset(Dataset):
    """
    PyTorch dataset for inference.

    Parameters
    ----------
    paths : Sequence[str]
        List of file paths to test images.
    transforms : A.BaseCompose
        Albumentations Compose with image augmentations.
    dim : int
        Patch size.
    stride : int
        Stride for patch extraction.
    padding : str, "reflect" or "constant"
        Type of padding for patch extraction.
    """

    def __init__(
        self,
        paths: Sequence[str],
        transforms: A.BaseCompose,
    ):
        self.paths = paths
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = cv2.imread(self.paths[index], cv2.IMREAD_GRAYSCALE)
        image = np.asarray(image, np.float32)

        image = self.transforms(image=image)["image"]
        image = (image - image.mean()) / (image.std() + 1e-6)

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
    paths : Sequence[str]
        Paths to images
    vessels : Sequence[str]
        Run-length encoded vessels masks
    kidney : Sequence[str]
        Run-length encoded kidney masks
    height : int
        Height of input images
    width : int
        Width of input images
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
