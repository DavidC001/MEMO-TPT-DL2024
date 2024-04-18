from torch.utils.data import Dataset
import cv2
import os
import csv

class ImageNetA(Dataset):
    """
    A custom dataset class for loading images from the ImageNet-A dataset.

    Args:
        root (str): The root directory of the dataset.
        csvMapFile (str, optional): The path to the CSV file containing the mapping of WordNet IDs to class names. Defaults to "dataloaders/wordNetIDs2Classes.csv".
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Defaults to None.
    """
    def __init__(self, root, csvMapFile="dataloaders/wordNetIDs2Classes.csv", transform=None):
        paths = []
        labels = []
        names = []

        mapping = {}
        csv_file = csv.reader(open(csvMapFile, 'r'))
        for id, wordnet, name in csv_file:
            if id == 'resnet_label':
                continue
            mapping[int(wordnet)] = {'id': id, 'name': name}

        #print(mapping)

        for classes in os.listdir(root):
            if classes == 'README.txt':
                continue
            for img in os.listdir(os.path.join(root, classes)):
                paths.append(os.path.join(root, classes, img).replace('\\', '/'))
                #remove n and leading 0s
                class_id = int(classes[1:])
                labels.append(mapping[class_id]['id'])
                names.append(mapping[class_id]['name'])

        self.data = {
            'paths': paths,
            'labels': labels,
            'names': names
        }
        self.transform = transform

    def getClassesNames(self):
        """
        Returns the class names of the dataset.
        """
        return set(self.data['names'])

    def __len__(self):
        return len(self.data['paths'])

    def __getitem__(self, idx):
        path = self.data['paths'][idx]
        label = self.data['labels'][idx]
        name = self.data['names'][idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return {
            'img': img,
            'label': label,
            'name': name
        }


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    root = 'datasets/imagenet-a'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImageNetA(root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(dataset.getClassesNames())
    

    for data in dataloader:
        img = data['img'][0].permute(1, 2, 0)
        label = data['label'][0]
        name = data['name'][0]

        plt.imshow(img)
        plt.title(f'{name} ({label})')
        plt.show()