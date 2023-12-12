from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import cleanup


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    scoring: Callable,
    device: torch.device,
    scheduler: Optional[LRScheduler] = None,
    accumulate: int = 1,
    threshold: float = 0.5,
    clipnorm: float = 1.0
) -> tuple[float, float]:
    """
    Train the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    loader : DataLoader
        PyTorch DataLoader containing training data.
    optimizer : Optimizer
        The optimizer to use for training.
    criterion : torch.nn.Module
        Loss function used for training.
    scoring : Callable
        Scoring function to evaluate model performance.
    device : torch.device
        The device where the computation should take place.
    scheduler : LRScheduler, optional
        Learning rate scheduler.
    accumulate : int, default=1
        Number of gradient accumulation steps
    threshold : float, default=0.5
        Threshold value to binarize prediction masks.
    clipnorm : float, default=1.0
        Gradients will be rescaled to have `clipnorm` vector norm.

    Returns
    -------
    tuple of floats
        A tuple containing the average loss and score over the training dataset.

    """
    model.train()
    loss, score = 0.0, 0.0
    scaler = torch.cuda.amp.GradScaler()
    pbar = tqdm(loader, total=len(loader), desc="Train")
    for step, batch in enumerate(pbar):
        batch = [element.to(device) for element in batch]
        with torch.autocast(device_type=str(device)):
            output = model.forward(batch[0])
            running_loss = criterion(output, *batch[1:])
            running_loss = running_loss / accumulate

        scaler.scale(running_loss).backward()
        if (step + 1) % accumulate == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), clipnorm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_score = scoring((output.sigmoid() > threshold).byte(), batch[1].byte())
        running_loss = running_loss * accumulate

        loss += running_loss.item()
        score += running_score[0].item()

        pbar.set_postfix(
            loss=running_loss.item(),
            vessel_score=running_score[0].item(),
            kidney_score=running_score[1].item(),
        )

    if hasattr(criterion, "step"):
        criterion.step()

    loss /= len(loader)
    score /= len(loader)

    cleanup()

    return loss, score
