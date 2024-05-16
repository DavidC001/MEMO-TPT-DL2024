import torchvision.transforms as transforms
from torchvision.transforms.v2 import AugMix
from torchvision.transforms import InterpolationMode


class EasyAgumenter(object):
    def __init__(self, base_transform, preprocess, augmentation, n_views=63):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views

        if augmentation == 'augmix':

            self.preaugment = transforms.Compose(
                [
                    AugMix(),
                    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                ]
            )
        elif augmentation == 'identity':
            self.preaugment = self.base_transform
        elif augmentation == 'cut':
            self.preaugment = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            raise ValueError('Augmentation type not recognized')
    
    def __call__(self, x):

        if isinstance(x, np.ndarray):
            x = transforms.ToPILImage()(x)

        image = self.preprocess(self.base_transform(x))

        views = [self.preprocess(self.preaugment(x)) for _ in range(self.n_views)]

        return [image] + views
