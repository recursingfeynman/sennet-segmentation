import numpy as np
import torch

from ._lookup_tables import create_table_neighbour_code_to_surface_area


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


def compute_area(y, unfold, area):
    yy = torch.stack(y, dim=0).to(torch.float16).unsqueeze(0)
    cubes_float = unfold(yy).squeeze(0)
    cubes_byte = torch.zeros(cubes_float.size(1), dtype=torch.int32)
    for k in range(8):
        cubes_byte += cubes_float[k, :].to(torch.int32) << k
    return area[cubes_byte]


def surface_dice(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute surface dice coefficient.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted volume [D, H, W]
    y_true : torch.Tensor
        Ground truth [D, H, W]

    Returns
    -------
    float
        Calculated surface dice coefficient

    References
    ----------
    [1] https://www.kaggle.com/code/junkoda/fast-surface-dice-computation/notebook.
    """
    area = create_table_neighbour_code_to_surface_area((1, 1, 1))
    unfold = torch.nn.Unfold(kernel_size=(2, 2), padding=1)

    D, H, W = y_true.shape
    y0 = y0_pred = torch.zeros((H, W), dtype=torch.uint8)
    intersection, union = 0, 0

    for i in range(D + 1):
        if i < D:
            y1, y1_pred = y_true[i], y_pred[i]
        else:
            y1 = y1_pred = torch.zeros((H, W), dtype=torch.uint8)

        area_pred = compute_area([y0_pred, y1_pred], unfold, area)
        area_true = compute_area([y0, y1], unfold, area)
        idx = torch.logical_and(area_pred > 0, area_true > 0)

        intersection += area_pred[idx].sum() + area_true[idx].sum()
        union += area_pred.sum() + area_true.sum()

        y0 = y1
        y0_pred = y1_pred

    dice = intersection / torch.clamp(union, 1e-6)
    return dice.item()


def _aggregate(score: torch.Tensor, reduction: bool) -> torch.Tensor:
    if reduction:
        score = torch.mean(score)

    return score
