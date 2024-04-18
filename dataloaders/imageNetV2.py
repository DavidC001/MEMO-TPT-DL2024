from torch.utils.data import Dataset
import cv2
import os
import csv

class ImageNetV2(Dataset):
    def __init__(self, root, csvMapFile="dataloaders/wordNetIDs2Classes.csv",  transform=None):
        paths = []
        labels = []
        names = []

        mapping = {}
        csv_file = csv.reader(open(csvMapFile, 'r'))
        for id, _, name in csv_file:
            if id == 'resnet_label':
                continue
            mapping[id] = name

        for classes in os.listdir(root):
            for img in os.listdir(os.path.join(root, classes)):
                paths.append(os.path.join(root, classes, img).replace('\\', '/'))
                labels.append(classes)
                names.append(mapping[classes])

        self.data = {
            'paths': paths,
            'labels': labels,
            'names': names
        }
        self.transform = transform

    def getClassesNames(self):
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

if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    root = 'datasets/imagenetv2-matched-frequency-format-val'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImageNetV2(root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for data in dataloader:
        img = data['img'][0].permute(1, 2, 0)
        label = data['label'][0]
        name = data['name'][0]

        plt.imshow(img)
        plt.title(f'{name} ({label})')
        plt.show()