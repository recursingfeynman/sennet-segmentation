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

    Returns
    -------
    tuple of floats
        A tuple containing the average loss and score over the evaluation dataset.

    """
    model.eval()
    loss, score = 0.0, 0.0
    pbar = tqdm(loader, total=len(loader), desc="Evaluation")
    for batch in pbar:
        batch = [b.to(device) for b in batch]
        with torch.autocast(device_type=str(device)):
            output = model.forward(batch[0])
            running_loss = criterion(output, *batch[1:])

        running_score = scoring(output.sigmoid(), batch[1])
        loss += running_loss.item()
        score += running_score.item()

        pbar.set_postfix(loss=running_loss.item(), score=running_score.item())

    loss /= len(loader)
    score /= len(loader)

    cleanup()

    return loss, score
