from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import cleanup


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    scoring: Callable,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """
    Evaluate model performance.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    loader : DataLoader
        PyTorch DataLoader containing test data.
    criterion : nn.Module
        Loss function used for evaluation.
    scoring : Callable
        Scoring function to evaluate model performance.
    device : torch.device
        The device where the computation should take place.
    threshold : float, default=0.5
        Threshold value to binarize prediction masks.

    Returns
    -------
    tuple of floats
        A tuple containing the average loss and score over the evaluation dataset.

    """
    model.eval()
    loss, score = 0.0, 0.0
    pbar = tqdm(loader, total=len(loader), desc="Evaluation")
    for images, masks, dtms in pbar:
        images = images.to(device)
        masks = masks.to(device)
        dtms = dtms.to(device)

        with torch.autocast(device_type=str(device)):
            output = model.forward(images)
            running_loss = criterion(output, masks, dtms)

        running_score = scoring((output.sigmoid() > threshold).byte(), masks.byte())

        loss += running_loss.item()
        score += running_score[0].item()

        pbar.set_postfix(
            loss=running_loss.item(),
            vessel_score=running_score[0].item(),
            kidney_score=running_score[1].item(),
        )

    loss /= len(loader)
    score /= len(loader)

    cleanup()

    return loss, score
