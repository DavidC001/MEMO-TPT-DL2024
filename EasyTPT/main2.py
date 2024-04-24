import sys

sys.path.append(".")

import os
import random
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from random import choice

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from dataloaders.imageNetA import ImageNetA
from dataloaders.imageNetV2 import ImageNetV2

from models import EasyTPT


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(
        dim=-1, keepdim=True
    )  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(
        logits.shape[0]
    )  # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def test_time_tuning(model, inputs, optimizer):
    model.eval()
    selected_idx = None
    for j in range(5):
        # with torch.cuda.amp.autocast():
        # print(f"[TTT] Iteration {j}")
        # print("NaN1? ", model.prompt_learner.emb_prefix[0][0][0].isnan().item())

        output = model(inputs)

        loss = avg_entropy(output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # breakpoint()

    # scaler.scale(loss).backward()
    # # Unscales the gradients of optimizer's assigned params in-place
    # scaler.step(optimizer)
    # scaler.update()

    return


device = "cuda:0"

tpt = EasyTPT(device)


for name, param in tpt.named_parameters():
    param.requires_grad_(False)


if not torch.cuda.is_available():
    print("Using CPU this is no bueno")
else:
    print("Switching to GPU, brace yourself!")
    torch.cuda.set_device(device)
    tpt = tpt.cuda(device)

transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()]
)
ima_root = "datasets/imagenet-a"
dataset = ImageNetA(ima_root, transform=transform)

# imv_root = "datasets/imagenetv2-matched-frequency-format-val"
# dataset = ImageNetV2(imv_root, transform=transform)

all_classes, mapping = dataset.getClassesNames()

breakpoint()

# NCLASSES = 200
NAUG = 63
NSAMPLES = 2000

# idxs = [random.randint(0, len(dataset) - 1) for _ in range(NSAMPLES)]
# # idxs = [i for i in range(NCLASSES)]
# elements = [dataset[idx] for idx in idxs]
# images = [element["img"] for element in elements]
# labels = [element["label"] for element in elements]
# classnames = [element["name"] for element in elements]

# prep = [tpt.preprocess(image) for image in images]
trans = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
)
tpt.prompt_learner.prepare_prompts(all_classes)

# setup automatic mixed-precision (Amp) loss scaling
scaler = torch.cuda.amp.GradScaler(init_scale=1000)

print("=> Using native Torch AMP. Training in mixed precision.")

import cv2 as cv

cv.namedWindow("image", cv.WINDOW_NORMAL)

LR = 0.005
trainable_param = tpt.prompt_learner.parameters()
optimizer = torch.optim.AdamW(trainable_param, LR)
optim_state = deepcopy(optimizer.state_dict())

correct = 0

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
for dat in dataloader:
    img = dat["img"]
    label = int(dat["label"][0])
    name = dat["name"][0]

    augs = [img] + [trans(img) for _ in range(NAUG)]
    prep_imgs = torch.tensor(np.stack(augs)).squeeze(1).cuda()

    # for name, param in tpt.named_parameters():
    #     if "prompt_learner" in name:
    #         print("LEARNABLE: ", name)

    tpt.eval()

    with torch.no_grad():
        tpt.reset()

    optimizer.load_state_dict(optim_state)
    test_time_tuning(tpt, prep_imgs, optimizer)
    with torch.no_grad():
        out = tpt(img.cuda())
    out_id = out.argmax(1).item()

    breakpoint()
    if out_id == i:
        correct += 1

    print(f"Predicted: {classnames[out_id]}\nTarget: {classnames[i]}")
    breakpoint()
    acc = correct / (i + 1)
    print(f"Accuracy: {acc}")
    # plt.imshow(out.cpu().detach().numpy(), cmap="hot", interpolation="nearest")
    # plt.colorbar()
    # plt.show()
    og_img = cv.cvtColor(np.array(images[i]), cv.COLOR_RGB2BGR)
    cv.imshow("image", cv.cvtColor(np.array(images[i]), cv.COLOR_RGB2BGR))
    cv.waitKey(1)
for i, label in enumerate(labels):
    print(f"Image {i} belongs to class {label}")

# Plot the scores


breakpoint()
