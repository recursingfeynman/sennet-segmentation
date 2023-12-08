import torch.nn as nn
import torch

class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.axes = (2, 3)

    def compute_weights(self, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates class weights from ground truth.

        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth.

        Returns
        -------
        torch.Tensor
            Computed class weights
        """
        # Class weights not defined for empty ground truth
        class_counts = torch.sum(y_true, (0, 2, 3))
        class_weights = 1.0 / class_counts / torch.sum(1.0 / class_counts)

        return class_weights