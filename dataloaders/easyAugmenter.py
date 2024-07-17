import torchvision.transforms as transforms
from torchvision.transforms.v2 import AugMix
from torchvision.transforms import InterpolationMode
import numpy as np


class EasyAgumenter(object):
    def __init__(self, base_transform, preprocess, augmentation, n_views=63):
        """
        This class provides an easy way to apply custom augmentations to images, the when called
        it will return a list of augmentations with the original image in first place.

        Args:

        - base_transform (torchvision.transforms.Compose): The base transformation to apply to the images.
        - preprocess (torchvision.transforms.Compose): The preprocessing transformation to apply to the images (will be applied last).
        - augmentation (str): The type of augmentation to apply, can be 'augmix', 'identity' or 'cut'.
        - n_views (int): The number of augmentations to apply to the image.

        Returns:
        - (list) A list of images with the augmentations applied.
        """
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views

        if augmentation == "augmix":

            self.preaugment = transforms.Compose(
                [
                    AugMix(),
                    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                ]
            )
        elif augmentation == "identity":
            self.preaugment = self.base_transform
        elif augmentation == "cut":
            self.preaugment = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            raise ValueError("Augmentation type not recognized")

    def __call__(self, x):

        if isinstance(x, np.ndarray):
            x = transforms.ToPILImage()(x)

        image = self.preprocess(self.base_transform(x))

        views = [self.preprocess(self.preaugment(x)) for _ in range(self.n_views)]

        return [image] + views
