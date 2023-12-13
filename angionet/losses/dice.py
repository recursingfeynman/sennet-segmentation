from typing import Optional

import torch
import torch.nn.functional as F

from ._base import BaseLoss


class DiceLoss(BaseLoss):
    """
    Dice Loss for semantic segmentation tasks.

    Parameters
    ----------
    sigmoid : bool, default=True
        Whether apply sigmoid function or not.
    class_weights : list, optional
        Weights for classes. If 'per_batch', it calculates weights
        per batch based on class frequencies. If 'None', weights equals 1.0.

    Attributes
    ----------
    sigmoid : bool
        Flag indicating whether apply sigmoid function to predictions or not.
    class_weights : torch.Tensor
        Weights for classes.
    axes : tuple
        Axes for reduction in loss calculations.

    Methods
    -------
    forward(y_pred, y_true)
        Calculate the Dice Loss between the predicted mask and target labels.

    """

    def __init__(
        self, sigmoid: bool = True, class_weights: Optional[list] = None
    ):
        super().__init__()
        self.sigmoid = sigmoid
        self.class_weights = class_weights

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, *args: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the Dice Loss between the predicted mask and target labels.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model outputs.
        y_true : torch.Tensor
            Target labels.
        args : torch.Tensor
            Exists due to compatibility.

        Returns
        -------
        torch.Tensor
            Computed dice loss.
        """
        if self.sigmoid:
            y_pred = F.logsigmoid(y_pred).exp()

        if self.class_weights == "per_batch":
            class_weights = self.compute_weights(y_true)
        elif isinstance(self.class_weights, list):
            class_weights = torch.tensor(self.class_weights, device=y_true.device)
        else:
            class_weights = torch.ones(y_true.size(1), device=y_true.device)

        intersection = torch.sum(y_pred * y_true, self.axes)
        intersection = torch.sum(intersection * class_weights, 1)

        union = torch.sum(y_pred, self.axes) + torch.sum(y_true, self.axes)
        union = torch.sum(union * class_weights, 1)

        return 1 - torch.mean((2 * intersection + 1e-6) / (union + 1e-6))
