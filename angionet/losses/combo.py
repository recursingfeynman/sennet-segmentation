from typing import Optional

import torch
import torch.nn as nn


class ComboLoss(nn.Module):
    """
    Combines multiple loss functions.

    Parameters
    ----------
    criterions : list of nn.Module
        A list of loss function instances to be combined.
    weights : list, optional
        A list of weights corresponding to each criterion. If None, weights are set
        to 1.0 for each criterion.

    Attributes
    ----------
    criterions : list of nn.Module
        List of loss functions.
    weights : list
        List with weights for each criterion.

    Methods
    -------
    forward(y_pred, y_true, *args)
        Computes the combined loss.
    """

    def __init__(self, criterions: list[nn.Module], weights: Optional[list] = None):
        super().__init__()
        self.criterions = criterions

        if weights is None:
            self.weights = [1.0] * len(criterions)
        else:
            self.weights = weights

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, *args: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the combined loss based on the provided criteria and weights.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted output tensor.
        y_true : torch.Tensor
            The target tensor.
        *args : torch.Tensor
            Extra tensors required by the specific loss function.

        Returns
        -------
        torch.Tensor
            The combined loss.
        """
        loss = torch.tensor([0.0], requires_grad=True, device=y_true.device)
        for criterion, weight in zip(self.criterions, self.weights):
            loss = loss + criterion(y_pred, y_true, *args) * weight
        return loss
