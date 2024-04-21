import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import data.augmix_ops as augmentations

ID_to_DIRNAME = {
    "I": "ImageNet",
    "A": "imagenet-a",
    "K": "ImageNet-Sketch",
    "R": "imagenet-r",
    "V": "imagenetv2-matched-frequency-format-val",
    "flower102": "Flower102",
    "dtd": "DTD",
    "pets": "OxfordPets",
    "cars": "StanfordCars",
    "ucf101": "UCF101",
    "caltech101": "Caltech101",
    "food101": "Food101",
    "sun397": "SUN397",
    "aircraft": "fgvc_aircraft",
    "eurosat": "eurosat",
}
from typing import Any


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


def build_dataset(
    set_id,
    transform,
    data_root,
):

    testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
    testset = DatasetWrapper(testdir, transform=transform)

    return testset


# AugMix Transforms
def get_preaugment():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
    )


def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity

    def __call__(self, x):

        image = self.preprocess(self.base_transform(x))
        views = [
            augmix(x, self.preprocess, self.aug_list, self.severity)
            for _ in range(self.n_views)
        ]
        return [image] + views
