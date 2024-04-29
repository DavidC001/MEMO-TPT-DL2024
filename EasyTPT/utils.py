import os

from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

import datasets
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode
from torchvision import transforms

from typing import Any, Tuple


class DatasetWrapper(datasets.ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        This function is overriden to return the path of the image as well

        Args:
            index (int): Index

        Returns:
            tuple: (sample, (target,path)) where target is class_index of the target class nad path the path to the image
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, (target, path)


class EasyAgumenter(object):
    def __init__(self, base_transform, preprocess, n_views=63):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.preaugment = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
        )

    def __call__(self, x):

        if isinstance(x, np.ndarray):
            x = transforms.ToPILImage()(x)

        image = self.preprocess(self.base_transform(x))

        views = [self.preprocess(self.preaugment(x)) for _ in range(self.n_views)]

        return [image] + views


def get_transforms(augs=64):

    base_transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
    )

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    data_transform = EasyAgumenter(
        base_transform,
        preprocess,
        n_views=augs - 1,
    )

    return data_transform
