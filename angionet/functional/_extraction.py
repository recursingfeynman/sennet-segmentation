import math

import numpy as np
import torch
import torch.nn.functional as F


def extract_patches(
    image: torch.Tensor, dim: int, stride: int, padding: str = "constant"
) -> torch.Tensor:
    """
    Extract patches from a batch of images.

    Parameters
    ----------
    image : torch.Tensor
        Batch of source images [B, C, H, W].
    dim : int
        The window size.
    stride : int
        The stride used for patch extraction.
    padding : str, default='constant'
        Padding method for images.

    Returns
    -------
    torch.Tensor
        Batch of extracted patches [B, N, C, H, W].

    Notes
    -----
    Copy from:
    https://github.com/kornia/kornia/blob/master/kornia/contrib/extract_patches.py
    """
    assert image.dim() == 4, "Input images must be [B, C, H, W]."

    B, C = image.shape[:2]
    pad = calc_pad(image.shape, dim, stride)
    image = F.pad(image, pad, mode=padding)
    axes = range(2, image.dim())
    for axis in axes:
        image = image.unfold(axis, dim, stride)
    image = image.permute(0, *axes, 1, 4, 5)
    return image.contiguous().view(B, -1, C, dim, dim)


def combine_patches(
    shape: tuple[int, ...],
    patches: torch.Tensor,
    dim: int,
    stride: int,
    lomc: bool = False,
) -> torch.Tensor:
    """
    Reconstructs original images from a batch of extracted patches.

    Parameters
    ----------
    shape : tuple of ints
        Original image size.
    patches : torch.Tensor
        Batch of extracted patches with shape [B, N, C, H, W].
    dim : int
        Window size during patch extraction.
    stride : int
        Stride during patch extraction.
    lomc : bool, default=False
        Logical OR mask combination (LOMC). By default, patches are merged
        with the `=` operator. LOMC includes both values ​​in the overlapping region.
        Only useful if the stride < dim. Patches should have appropriate
        dtype (e.g. uint8).

    Returns
    -------
    torch.Tensor
        Batch of reconstructed images [B, C, H, W].
    """
    assert patches.dim() == 5, "Patches should be [B, N, C, H, W]."

    B, N, C, _, _ = patches.shape
    left, right, top, bottom = calc_pad(shape, dim, stride)
    padded_shape = shape[-2] + top + bottom, shape[-1] + left + right
    locs = calc_locs(padded_shape, dim, stride)

    rec = torch.empty((B, C, *padded_shape)).type_as(patches)
    for n in range(N):
        ypos, xpos = locs[n, 0] * stride, locs[n, 1] * stride

        if lomc:
            rec[..., ypos : ypos + dim, xpos : xpos + dim] |= patches[:, n]
        else:
            rec[..., ypos : ypos + dim, xpos : xpos + dim] = patches[:, n]

    return rec[..., top : shape[-2] + top, left : shape[-1] + left]


def calc_pad(shape: tuple[int, ...], dim: int, stride: int) -> tuple[int, ...]:
    padh = math.ceil((shape[-2] - dim) / stride) * stride - (shape[-2] - dim)
    padw = math.ceil((shape[-1] - dim) / stride) * stride - (shape[-1] - dim)

    if padh == shape[-2]:
        top, bottom = 0, 0
    else:
        top, bottom = padh // 2, padh - padh // 2

    if padw == shape[-1]:
        left, right = 0, 0
    else:
        left, right = padw // 2, padw - padw // 2

    return left, right, top, bottom


def calc_locs(shape: tuple[int, ...], dim: int, stride: int) -> np.ndarray:
    ytot = math.ceil((shape[-2] - dim + 1) / stride)
    xtot = math.ceil((shape[-1] - dim + 1) / stride)
    ys, xs = np.meshgrid(np.arange(ytot), np.arange(xtot), indexing="ij")

    return np.stack((ys.flatten(), xs.flatten()), axis=1)
