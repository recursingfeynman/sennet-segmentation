import math
from typing import Optional

import torch
import torch.nn.functional as F

from ._base import BaseLoss


class GenSurfLoss(BaseLoss):
    """
    Generalized Surface Loss for semantic segmentation tasks.
    Paper: https://arxiv.org/pdf/2302.03868.pdf

    Parameters
    ----------
    region_loss : torch.nn.Module
        The region-based loss function.
    total_steps : int
        Total training steps for alpha decay.
    sigmoid : bool
        Wheter apply sigmoid function or not.
    class_weights : torch.Tensor, optional
        Weights for different classes. If 'per_batch', it calculates weights
        per batch based on class frequencies. If 'None', weights equals 1.0.

    Attributes
    ----------
    region_loss : torch.nn.Module
        The region-based loss. Should accepts logits.
    total_steps : int
        Total training steps for alpha decay.
    sigmoid : bool
        Flag indicating whether apply sigmoid or not.
    class_weights : torch.Tensor
        Weights for classes.
    axes : tuple
        Axes for sum reduction in loss calculations.
    iter : int
        Current iteration count for alpha decay.
    alpha : float
        Alpha parameter for balancing region and boundary losses.

    Methods
    -------
    step()
        Update the iteration count and calculate alpha based on cosine scheduling.

    forward(y_pred, y_true, dtms)
        Compute the Generalized Surface Loss.

    """

    def __init__(
        self,
        region_loss: torch.nn.Module,
        total_steps: int,
        sigmoid: bool = True,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.region_loss = region_loss
        self.total_steps = total_steps - 1
        self.sigmoid = sigmoid
        self.class_weights = class_weights
        self.iter = 0
        self.alpha = 1.0

    def step(self):
        """
        Update the iteration count and calculate alpha for alpha decay.
        """
        self.iter = self.iter + 1
        self.alpha = 0.5 * (1 + math.cos(math.pi * self.iter / self.total_steps))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        dtms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Generalized Surface Loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model output.
        y_true : torch.Tensor
            Ground truth labels.
        dtms : torch.Tensor
            Distance transforms maps (DTM) for each class. Values in the DTM have to be
            positive on the exterior, zero on the boundary, and negative in the
            object`s interior.

        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        region_loss = self.region_loss(y_pred, y_true)

        if self.sigmoid:
            y_pred = F.logsigmoid(y_pred).exp()

        if self.class_weights == "per_batch":
            class_weights = self.compute_weights(y_true)
        elif self.class_weights is not None:
            class_weights = self.class_weights
        else:
            class_weights = torch.tensor([1.0, 1.0], device=y_true.device)

        num = torch.sum((dtms * (1 - (y_true + y_pred))) ** 2, self.axes)
        num = torch.sum(num * class_weights, 1)

        den = torch.sum(dtms**2, self.axes)
        den = torch.sum(den * class_weights, 1)

        boundary_loss = 1 - torch.mean(num / den)

        return self.alpha * region_loss + (1.0 - self.alpha) * boundary_loss
