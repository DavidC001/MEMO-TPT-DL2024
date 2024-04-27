import os

from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

import datasets
import torchvision.datasets as datasets


from typing import Any, Tuple


class DatasetWrapper(datasets.ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
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

        image = self.preprocess(self.base_transform(x))

        views = [self.preprocess(self.preaugment(x)) for _ in range(self.n_views)]
        return [image] + views
