import torchvision.transforms as transforms
from torchvision.transforms.v2 import AugMix
from torchvision.transforms import InterpolationMode


class EasyAgumenter(object):
    def __init__(self, base_transform, preprocess, augmix, n_views=63):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views

        if augmix=='augmix':

            self.preaugment = transforms.Compose(
                [
                    AugMix(),
                    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                ]
            )
        elif augmix=='identity':
            self.preaugment = self.base_transform
        else:
            self.preaugment = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                ]
            )