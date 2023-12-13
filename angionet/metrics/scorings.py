import numpy as np
import torch

from ._surface_distance import (
    compute_surface_dice_at_tolerance,
    compute_surface_distances,
)


def confusion_matrix(y_pred: torch.Tensor, y_true: torch.Tensor) -> np.ndarray:
    """
    Compute confusion matrix values.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted binary mask.
    y_true : torch.Tensor
        Ground truth.

    Returns
    -------
    np.array
        Computed confusion matrix values: tn, fp, fn, tp.

    References
    ----------
    [1] https://stackoverflow.com/questions/59080843/
    """
    y_pred = y_pred.flatten().long()
    y_true = y_true.flatten().long()
    N = max(max(y_true), max(y_pred)) + 1
    y = N * y_true + y_pred
    y = torch.bincount(y)
    if len(y) < N * N:
        y = torch.cat((y, torch.zeros(N * N - len(y), dtype=torch.long)))

    return y.numpy()


def dice(y_pred, y_true, reduction=True):
    """
    Calculate Sørensen-Dice coefficient from probabilities.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted mask [0, 1].
    y_true : torch.Tensor
        Ground truth.
    reduction : bool, default=True
        Average of calculated scores

    Returns
    -------
    torch.Tensor
        Dice similarity coefficient.
    """
    if (y_pred.dim() < 4) or (y_true.dim() < 4):
        raise ValueError("Only BxCxHxW input supported.")

    intersection = (y_pred * y_true).sum((1, 2, 3))
    union = (y_pred**2 + y_true**2).sum((1, 2, 3))
    score = (2 * intersection + 1e-6) / (union + 1e-6)
    return _aggregate(score, reduction)


def jaccard(
    y_pred: torch.Tensor, y_true: torch.Tensor, reduction: bool = True
) -> torch.Tensor:
    """
    Calculate Jaccard similarity coefficient (IoU).

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted binary mask.
    y_true : torch.Tensor
        Ground truth.
    reduction : bool, default=True
        Average of calculated scores

    Returns
    -------
    torch.Tensor
        Jaccard similarity coefficient.
    """
    if (y_pred.dim() < 4) or (y_true.dim() < 4):
        raise ValueError("Only BxCxHxW input supported.")

    intersection = (y_pred & y_true).sum((0, 2, 3)).float()
    union = (y_pred | y_true).sum((0, 2, 3)).float()
    score = (intersection + 1e-6) / (union + 1e-6)

    return _aggregate(score, reduction)


def surface_dice(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    tolerance: float = 0.0,
    spacing: tuple[int, ...] = (1, 1),
) -> float:
    """
    Compute the surface Dice coefficient at a specified tolerance.

    See https://github.com/google-deepmind/surface-distance.

    Parameters
    ----------
    y_pred : np.array
        Predicted binary mask {0, 1}.
    y_true : np.array
        Ground truth.
    tolerance : float, default=0.0
        The tolerance in mm.
    spacing : tuple[int, ...], default=(1, 1)
        Voxel spacing in x0 anx x1 (resp. x0, x1 and x2) directions.

    Returns
    -------
    float
        Surface Dice coefficient.
    """
    assert y_true.ndim == 4 or y_pred.ndim == 4, "Only BxCxHxW input supported."

    y_pred = np.asarray(y_pred, dtype="bool")
    y_true = np.asarray(y_true, dtype="bool")

    if len(spacing) == 2:
        B, C = y_true.shape[:2]
        scores = np.empty((B, C))
        for b in range(B):
            for c in range(C):
                dists = compute_surface_distances(y_pred[b, c], y_true[b, c], spacing)
                scores[b, c] = compute_surface_dice_at_tolerance(dists, tolerance)
        scores = np.mean(scores, axis=0)
    elif len(spacing) == 3:
        B = y_true.shape[0]
        scores = np.empty((B,))
        for b in range(B):
            dists = compute_surface_distances(y_pred[b], y_true[b], spacing)
            scores[b] = compute_surface_dice_at_tolerance(dists, tolerance)
        scores = np.mean(scores)
    else:
        raise ValueError("Only 2- or 3-dim spacing supported.")

    return scores


def _aggregate(score: torch.Tensor, reduction: bool) -> torch.Tensor:
    if reduction:
        score = torch.mean(score)

    return score
