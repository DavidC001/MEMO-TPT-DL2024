import torchvision.transforms as transforms
import sys

sys.path.append('.')
from dataloaders.dataloader import get_dataloaders
from dataloaders.easyAugmenter import EasyAgumenter


def memo_get_datasets(augmentation, augs=64):
    """
    Returns the ImageNetA and ImageNetV2 datasets for the memo model
    Args:
        augmentation (str): What type of augmentation to use in EasyAugmenter. Can be 'augmix', 'identity' or 'cut'
        augs (int): The number of augmentations to compute. Must be greater than 1

    Returns: The ImageNetA and ImageNetV2 datasets for the memo model, with the Augmentations already applied

    """
    assert augs > 1, 'The number of augmentations must be greater than 1'
    memo_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = EasyAgumenter(memo_transforms, preprocess, augmentation, augs - 1)
    imageNet_A, imageNet_V2 = get_dataloaders('datasets', transform)
    return imageNet_A, imageNet_V2
