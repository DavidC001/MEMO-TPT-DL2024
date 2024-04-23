import torch
import torch.nn as nn
from torchvision.transforms import v2
from matplotlib import pyplot as plt
import sys

sys.path.append('.')

from memo.utils.train_helpers import te_transforms
# from memo.utils.third_party import indices_in_1k, imagenet_r_mask


def adapt_single(model, image, optimizer, criterion, niter, batch_size, prior_strength):
    model.eval()
    # Using AugMix augmentation provided directly by pytorch instead of the things done by the paper
    augmenter = v2.AugMix()
    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)

    for iteration in range(niter):
        inputs = [augmenter(image) for _ in range(batch_size)]
        inputs = torch.stack(inputs).cuda()
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
        outputs = model(inputs.cuda())
        _, predicted = outputs.max(1)
        confidence = nn.functional.softmax(outputs, dim=1).squeeze()[predicted].item()
    correctness = 1 if predicted.item() == label else 0
    nn.BatchNorm2d.prior = 1
    return correctness, confidence
