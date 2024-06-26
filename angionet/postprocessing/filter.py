import cc3d
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import apply_hysteresis_threshold


def apply_threshold(
    volume: np.ndarray, lo: float, hi: float, chunk: int = 32
) -> np.ndarray:
    D, H, W = volume.shape
    predict = np.zeros((D, H, W), dtype="uint8")
    for i in range(0, D, chunk):
        predict[i : i + chunk] = np.maximum(
            apply_hysteresis_threshold(volume[i : i + chunk], lo, hi),
            predict[i : i + chunk],
        )
    return predict


def fill_holes(volume: np.ndarray, chunk: int = 32) -> np.ndarray:
    D, H, W = volume.shape
    predict = np.zeros((D, H, W), dtype="bool")
    for i in range(0, D, chunk):
        predict[i : i + chunk] = binary_fill_holes(volume[i : i + chunk])

    return np.asarray(predict, dtype="uint8")


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


def remove_dust(volume: np.ndarray, threshold: int, connectivity: int) -> np.ndarray:
    """
    Remove dust from input volume.

    Parameters
    ----------
    volume : np.array
        Predicted volume.
    threshold : int
        An integer value used as the threshold for connected components.
        Connected components with sizes less than this threshold will be removed.
    connectivity : int
        An integer specifying the connectivity of the connected components.
        It determines which neighboring pixels are considered connected.
        Only 4,8 (2D) and 26, 18, and 6 (3D) are allowed.

    """
    volume = cc3d.dust(volume, connectivity=connectivity, threshold=threshold)
    return volume
