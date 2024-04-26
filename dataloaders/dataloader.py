import sys
sys.path.append('.')
import os
import csv

from dataloaders.imageNetA import ImageNetA
from dataloaders.imageNetV2 import ImageNetV2

def get_dataloaders(root, transform=None):
    """
    Returns the dataloader of the dataset.

    Args:
        root (str): The root directory of the dataset.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Defaults to None.
    """
    root_A = os.path.join(root, "imagenet-a")
    imageNet_A = ImageNetA(root_A, transform=transform)
    root_V2 = os.path.join(root, "imagenetv2-matched-frequency-format-val")
    imageNet_V2 = ImageNetV2(root_V2, transform=transform)

    return imageNet_A, imageNet_V2

def get_classes_names(csvMapFile="dataloaders/wordNetIDs2Classes.csv"):
    """
    Returns the class names of the dataset.

    Args:
        csvMapFile (str, optional): The path to the CSV file containing the mapping of WordNet IDs to class names. Defaults to "dataloaders/wordNetIDs2Classes.csv".
    """
    names = [""]*1000
    csv_file = csv.reader(open(csvMapFile, 'r'))
    for id, wordnet, name in csv_file:
        if id == 'resnet_label':
            continue
        names[int(id)] = name
    
    return names


if __name__ == '__main__':    
    import matplotlib.pyplot as plt
    import random
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    imageA, imageV2 = get_dataloaders("datasets", transform=transform)

    #show 5 random images from each dataset
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("ImageNet-A & ImageNet-V2", fontsize=16)

    for i in range(5):
        idx = random.randint(0, len(imageA)-1)
        axs[0, i].imshow(imageA[idx]["img"].permute(1, 2, 0))
        axs[0, i].set_title(f"{imageA[idx]['name']} ({imageA[idx]['label']})")

        idx = random.randint(0, len(imageV2)-1)
        axs[1, i].imshow(imageV2[idx]["img"].permute(1, 2, 0))
        axs[1, i].set_title(f"{imageV2[idx]['name']} ({imageV2[idx]['label']})")
    
    plt.show()

    print("Classes names:")
    print(get_classes_names())

    print("Done!")