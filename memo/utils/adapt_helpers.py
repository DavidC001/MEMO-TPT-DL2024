import torch
import torch.nn as nn
from torchvision.transforms import v2
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import sys


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

sys.path.append('.')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
te_transforms = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize])

def adapt_single(model, image, optimizer, criterion, niter, batch_size, prior_strength):
    model.eval()
    # Using AugMix augmentation provided directly by pytorch
    augmenter = v2.AugMix()
    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)

    for iteration in range(niter):
        inputs = [augmenter(image) for _ in range(batch_size)]
        inputs = torch.stack(inputs).to(device=DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, logits = criterion(outputs)
        loss.backward()
        optimizer.step()
    nn.BatchNorm2d.prior = 1


def test_single(model, image, label, prior_strength):
    model.eval()

    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)
    transform = te_transforms
    inputs = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs.to(device=DEVICE))
        _, predicted = outputs.max(1)
        confidence = nn.functional.softmax(outputs, dim=1).squeeze()[predicted].item()
    correctness = 1 if predicted.item() == label else 0
    nn.BatchNorm2d.prior = 1
    return correctness, confidence
