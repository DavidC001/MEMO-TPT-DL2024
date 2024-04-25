import torch
import torch.nn as nn
from torchvision.transforms import v2
import torchvision.transforms as transforms

import sys
sys.path.append('.')
from dataloaders.dataloader import get_classes_names

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
te_transforms = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize])


def adapt_single(model, image, optimizer, criterion, niter, batch_size, prior_strength, device):
    model.eval()
    # Using AugMix augmentation provided directly by pytorch
    augmenter = v2.AugMix()

    nn.BatchNorm2d.prior = prior_strength

    for iteration in range(niter):
        inputs = [augmenter(image) for _ in range(batch_size)]
        inputs = torch.stack(inputs).to(device=device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, logits = criterion(outputs)
        loss.backward()
        optimizer.step()
    nn.BatchNorm2d.prior = 1

names = get_classes_names()

def test_single(model, image, label, prior_strength, device):
    model.eval()
    nn.BatchNorm2d.prior = prior_strength

    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image.to(device=device))
        _, predicted = outputs.max(1)
        #print( "Predicted: ", names[predicted.item()], " Label: ", names[label])
        confidence = nn.functional.softmax(outputs, dim=1).squeeze()[predicted].item()
    correctness = 1 if predicted.item() == label else 0
    nn.BatchNorm2d.prior = 1
    return correctness, confidence
