from torch.utils.data import Dataset
import cv2
import os

class ImageNetA(Dataset):
    def __init__(self, root, transform=None):
        paths = []
        labels = []
        names = []

        mapping = {}
        csv_wordNet2ImageNet = 'wordNetIDs2Classes.csv'
        with open(csv_wordNet2ImageNet, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                id, wordnet, name = line.strip().split(',')
                mapping[wordnet] = {'id': id, 'name': name}

        for classes in os.listdir(root):
            for img in os.listdir(os.path.join(root, classes)):
                paths.append(os.path.join(root, classes, img).replace('\\', '/'))
                labels.append(mapping[classes[2:]]['id'])
                names.append(mapping[classes[2:]]['name'])

        self.data = {
            'paths': paths,
            'labels': labels,
            'names': names
        }
        self.transform = transform

    def getClasses(self):
        return self.data['names']

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
