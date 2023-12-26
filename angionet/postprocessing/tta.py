import random
from typing import Callable, Optional

import torch
import torchvision.transforms.functional as F

from ._base import BaseTransform


class Compose(BaseTransform):
    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def augment(self, image: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            image = t.augment(image)
        return image

    def disaugment(self, image: torch.Tensor) -> torch.Tensor:
        for t in self.transforms[::-1]:
            image = t.disaugment(image)
        return image


class Combine(BaseTransform):
    def __init__(self, transforms: list[Compose | BaseTransform]):
        self.transforms = transforms

    def augment(self, image: torch.Tensor) -> torch.Tensor:
        augs = []
        for t in self.transforms:
            augs.append(t.augment(image))
        return torch.stack(augs)

    def disaugment(self, augs: torch.Tensor) -> torch.Tensor:
        images = []
        for index, t in enumerate(self.transforms):
            images.append(t.disaugment(augs[index]))
        return torch.stack(images)


class TransformWrapper(BaseTransform):
    def __init__(self, transform: Callable, inverse: Optional[Callable] = None):
        self.transform = transform
        self.inverse = inverse

    def augment(self, image: torch.Tensor) -> torch.Tensor:
        image = self.transform(image)
        return image

    def disaugment(self, image: torch.Tensor) -> torch.Tensor:
        if self.inverse is not None:
            image = self.inverse(image)
        return image


class Identity(BaseTransform):
    def augment(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def disaugment(self, image: torch.Tensor) -> torch.Tensor:
        return image


class HorizontalFlip(BaseTransform):
    def augment(self, image: torch.Tensor) -> torch.Tensor:
        image = F.hflip(image)
        return image

    def disaugment(self, image: torch.Tensor) -> torch.Tensor:
        image = F.hflip(image)
        return image


class VerticalFlip(BaseTransform):
    def augment(self, image: torch.Tensor) -> torch.Tensor:
        image = F.vflip(image)
        return image

    def disaugment(self, image: torch.Tensor) -> torch.Tensor:
        image = F.vflip(image)
        return image


class Rotate90(BaseTransform):
    def __init__(self, sign: Optional[int] = None):
        if sign is not None:
            self.angle = 90 * sign
        else:
            self.angle = 90 * random.choice((-1, 1))

    def augment(self, image: torch.Tensor) -> torch.Tensor:
        image = F.rotate(image, -self.angle)
        return image

    def disaugment(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        image = F.rotate(image, self.angle)
        return image
