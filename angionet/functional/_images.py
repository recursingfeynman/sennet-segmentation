from typing import Optional

import ants
import cv2
import numpy as np


def encode(mask: np.ndarray) -> str:
    """
    Encode a binary mask using Run-Length Encoding (RLE) and return the RLE string.

    Parameters
    ----------
    mask : np.array
        A binary mask represented as a NumPy array.

    Returns
    -------
    str
        The RLE-encoded string representing the input binary mask.
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = " ".join(str(x) for x in runs)
    if rle == "":
        rle = "1 0"
    return rle


def decode(rle: str, dim: tuple[int, ...]) -> np.ndarray:
    """
    Decode a Run-Length Encoded (RLE) string and return the corresponding binary mask.

    Parameters
    ----------
    rle : str
        The RLE-encoded string.
    dim : tuple[int, int]
        The size of the output binary mask.

    Returns
    -------
    np.array
        The binary mask represented as a NumPy array.
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(dim[0] * dim[1], dtype="uint8")
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(dim)


def cdist(mask: np.ndarray, dtype: Optional[str] = None) -> np.ndarray:
    """
    Calculates euclidean distance transform map.

    Parameters
    ----------
    mask : np.array
        Input binary mask.
    dtype : str, optional
        Output dtype.

    Returns
    -------
    np.array
        Calculated euclidean distance transform map.
    """
    mask = ants.from_numpy(mask.astype("float32"))
    dtm = ants.utils.iMath(mask, "MaurerDistance").numpy()

    if dtype is not None:
        dtm = np.asarray(dtm, dtype=dtype)

    return dtm


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Remove objects from input mask with size < min_size.

    Parameters
    ----------
    mask : np.array
        Input binary mask {0, 1}.
    min_size : int
        Min size of objects to remove.

    Returns
    -------
    np.array
        Processed binary mask.
    """
    mask = (mask * 255).astype(np.uint8)
    num_label, label, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    processed = np.zeros_like(mask)
    for cl in range(1, num_label):
        if stats[cl, cv2.CC_STAT_AREA] >= min_size:
            processed[label == cl] = 1
    return processed
