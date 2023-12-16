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
        device: str = "cpu",
    ):
        self.model = model
        self.transforms = transforms
        self.device = device

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        augs = []
        for transform in self.transforms:
            aug = transform.augment(image)
            with torch.autocast(device_type=image.device.type):
                aug = self.model.forward(aug)
            augs.append(transform.disaugment(aug.to(self.device)))
        return torch.stack(augs)


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
