from itertools import chain
from typing import Any, Callable, Optional, Sequence

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
    class_index : list of ints
        Class indices.
    dtms : bool
        Whether include distance transform maps in outputs or not.
    normalization : callable, optional
        The normalization function.
    stats : tuple of floats, optional
        Normalization statistics.
    """

    def __init__(
        self,
        paths: Sequence[str],
        transforms: A.BaseCompose,
        class_index: list[int] = [0],
        dtms: bool = False,
        normalization: Optional[Callable] = None,
        stats: Optional[Sequence[tuple]] = None,
    ):
        self.paths = paths
        self.transforms = transforms
        self.class_index = class_index
        self.dtms = dtms
        self.stats = stats
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sample = list(np.load(self.paths[index]).values())

        D = 2 if self.dtms else 1
        masks = np.stack(sample[1:], dtype = 'float32')[:D, self.class_index]
        masks = chain.from_iterable([unbind(m) for m in unbind(masks)])
        augs = self.transforms(image=sample[0].astype('float32'), masks=list(masks))

        masks = [torch.stack(augs["masks"][: len(self.class_index)])]
        if self.dtms:
            masks.append(torch.stack(augs["masks"][len(self.class_index) :]))

        if self.normalization is not None:
            stats = self.stats[index] if self.stats is not None else None
            image = self.normalization(augs["image"], stats)
        else:
            image = augs['image']

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
    stats : tuple of floats
        Normalization statistics.
    """

    def __init__(
        self,
        paths: Sequence[str],
        transforms: A.BaseCompose,
        normalize: Any = None,
        stats: Optional[tuple] = None,
    ):
        self.paths = paths
        self.transforms = transforms
        self.normalize = normalize if normalize is not None else rescale
        self.stats = stats

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = cv2.imread(self.paths[index], cv2.IMREAD_GRAYSCALE)
        image = np.asarray(image, np.float32)

        image = self.transforms(image=image)["image"]

        if self.stats is not None:
            image = self.normalize(image, self.stats[index])
        else:
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
