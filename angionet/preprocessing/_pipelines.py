from typing import Any

import cv2
import numpy as np
import torch

from ..functional import decode, cdist, extract_patches


def prepare_input(
    path: str, rles: list[str], config: Any
) -> list[tuple[str, int, int]]:
    """
    Prepares single path to convinient format for model training.
    Load image, decode RLE, extract patches, compute distance transform maps, save
    as .npz.

    Parameters
    ----------
    path : str
        Path to source image.
    rles : list[str]
        List of run-length encoded masks.
    config : Any
        Configuration class. Should have dim, stride, padding attributes.

    Returns
    -------
    list[tuple[str, int, int]]
        List of metadata from extracted patches (filepath, vessels pixels, kidney pixels).
    """
    dim = config.dim
    stride = config.stride
    padding = config.padding

    image = torch.from_numpy(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    masks = [torch.from_numpy(decode(rle, image.shape)) for rle in rles]
    batch = torch.stack((image, *masks)).unsqueeze(1).to(torch.float32)

    patches = extract_patches(batch, dim, stride, padding)
    patches = patches.squeeze().transpose(1, 0).numpy()  # [B, N, ...] -> [N, B, ...]

    data = []
    for index, patch in enumerate(patches):
        pixels_v = np.sum(patch[1])
        pixels_k = np.sum(patch[2])
        if pixels_v > 0:
            image = patch[0].astype("uint8")
            masks = patch[1:].astype("uint8")
            dtms = np.stack((cdist(patch[1]), cdist(patch[2])), dtype="float16")

            group = path.replace("sparse", "dense").split("/")[-3]
            image_id = path.split("/")[-1].replace(".tif", "")
            filename = f"{group}/{image_id}-{index}.npz"

            np.savez_compressed(file=filename, image=image, masks=masks, dtms=dtms)
            data.append((filename, pixels_v, pixels_k))

    return data
