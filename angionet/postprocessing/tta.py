import random
from typing import Callable, Optional

import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class TestTimeAugmentations:
    def __init__(
        self,
        model: nn.Module,
        transforms: list,
        aggregation: str | Callable = "mean",
        device: str = "cpu",
    ):
        self.model = model
        self.transforms = transforms
        self.aggregation = aggregation
        self.device = device

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        augs = []
        for transform in self.transforms:
            aug = transform.augment(image)
            with torch.autocast(device_type=str(image.device)):
                aug = self.model.forward(aug)
            augs.append(transform.disaugment(aug.to(self.device)))

        return self._aggregate(torch.stack(augs), self.aggregation)

    def _aggregate(self, augs: torch.Tensor, agg: str | Callable) -> torch.Tensor:
        if agg == "mean":
            augs = torch.mean(augs.sigmoid(), 0)
        elif agg == "max":
            augs = torch.max(augs.sigmoid(), 0).values
        elif callable(agg):
            augs = agg(augs.sigmoid())
        else:
            raise ValueError(f"{agg} not supported. Available: mean, max, callable.")
        return augs


class TransformWrapper:
    def __init__(self, transform: Callable, inverse: Optional[Callable] = None):
        self.transform = transform
        self.inverse = inverse

    def augment(self, image: torch.Tensor) -> torch.Tensor:
        image = self.transform(image)
        return image

    def disugment(self, image: torch.Tensor) -> torch.Tensor:
        if self.inverse is not None:
            image = self.inverse(image)
        return image


class HorizontalFlip:
    def augment(self, image: torch.Tensor) -> torch.Tensor:
        image = F.hflip(image)
        return image

    def disugment(self, image: torch.Tensor) -> torch.Tensor:
        image = F.hflip(image)
        return image


class VerticalFlip:
    def augment(self, image: torch.Tensor) -> torch.Tensor:
        image = F.vflip(image)
        return image

    def disugment(self, image: torch.Tensor) -> torch.Tensor:
        image = F.vflip(image)
        return image


class Rotate90:
    def __init__(self, sign: Optional[int] = None):
        if sign is not None:
            self.angle = 90 * sign
        else:
            self.angle = 90 * random.choice((-1, 1))

    def augment(self, image: torch.Tensor) -> torch.Tensor:
        image = F.rotate(image, -self.angle)
        return image

    def disugment(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        image = F.rotate(image, self.angle)
        return image
