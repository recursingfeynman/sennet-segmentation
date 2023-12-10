from typing import Optional

import numpy as np
import torch


def rescale(
    image: torch.Tensor | np.ndarray, stats: Optional[tuple[float, float]] = None
) -> torch.Tensor | np.ndarray:
    """
    Rescale input image using min-max normalization technique.

    Parameters
    ----------
    image : array-like
        Input image to rescale.
    stats : tuple of floats, optional
        Normalization statistics (min, max). If not specified, scaling will be
        done based on the minimum and maximum values of the input image.

    Returns
    -------
    array-like
        Rescaled image.
    """
    if stats is not None:
        image = (image - stats[0]) / (stats[1] - stats[0])
    else:
        image = (image - image.min()) / (image.max() - image.min())
    return image
