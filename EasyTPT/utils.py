import os

from PIL import Image
import numpy as np
import sys

import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode
from torchvision import transforms

from typing import Any, Tuple

from dataloaders.imageNetA import ImageNetA
from dataloaders.imageNetV2 import ImageNetV2

sys.path.append(".")
from EasyTPT.tpt_classnames.imagnet_prompts import imagenet_classes
from dataloaders.easyAugmenter import EasyAgumenter


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


def tpt_get_transforms(augs=64):

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


def tpt_get_datasets(data_root, augmix=False, augs=64, all_classes=True):
    """
    Returns the ImageNetA and ImageNetV2 datasets.

    Parameters:
    - data_root (str): The root directory of the datasets.
    - augmix (bool): Whether to use AugMix or not.
    - augs (int): The number of augmentations to use.
    - all_classes (bool): Whether to use all classes or not.

    Returns:
    - imageNet_A (ImageNetA): The ImageNetA dataset.
    - ima_names (list): The original classnames in ImageNetA.
    - ima_custom_names (list): The retouched  classnames in ImageNetA.
    - ima_id_mapping (list): The mapping between the index of the classname and the ImageNet label

    same for ImageNetV2

    For instance the first element of ima_names corresponds to the label '90'.  After running the
    inference run the predicted output through the ima_id_mapping to recover the correct class label.

    out = tpt(inputs)
    pred = out.argmax().item()
    out_id = ima_id_mapping[pred]

    """
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
        augmentation=("augmix" if augmix else "cut"),
        n_views=augs - 1,
    )

    imageNet_A = ImageNetA(
        os.path.join(data_root, "imagenet-a"), transform=data_transform
    )
    imageNet_V2 = ImageNetV2(
        os.path.join(data_root, "imagenetv2-matched-frequency-format-val"),
        transform=data_transform,
    )

    imv2_label_mapping = list(imageNet_V2.classnames.keys())
    imv2_names = list(imageNet_V2.classnames.values())
    imv2_custom_names = [imagenet_classes[int(i)] for i in imv2_label_mapping]

    ima_label_mapping = list(imageNet_A.classnames.keys())
    ima_names = list(imageNet_A.classnames.values())
    ima_custom_names = [imagenet_classes[int(i)] for i in ima_label_mapping]

    if all_classes:
        ima_names += [name for name in imv2_names if name not in ima_names]
        ima_custom_names += [
            name for name in imv2_custom_names if name not in ima_custom_names
        ]
        ima_label_mapping += [
            map for map in imv2_label_mapping if map not in ima_label_mapping
        ]

    return (
        imageNet_A,
        ima_names,
        ima_custom_names,
        ima_label_mapping,
        imageNet_V2,
        imv2_names,
        imv2_custom_names,
        imv2_label_mapping,
    )
