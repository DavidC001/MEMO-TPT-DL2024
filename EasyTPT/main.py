import sys

sys.path.append(".")

import os
import random
import math
import torch
import matplotlib.pyplot as plt
import numpy as np

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

    selected_idx = None
    for j in range(5):
        # with torch.cuda.amp.autocast():
        print(f"[TTT] Iteration {j}")
        print("NaN? ", model.prompt_learner.ctx[0][0][0].isnan().item())

        output = model(inputs)

        loss = avg_entropy(output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224))])
# dataset = ImageNetA("datasets/imagenet-a", transform=transform)
dataset = ImageNetV2(
    "datasets/imagenetv2-matched-frequency-format-val", transform=transform
)


idxs = [random.randint(0, len(dataset) - 1) for _ in range(10)]
elements = [dataset[idx] for idx in idxs]
images = [element["img"] for element in elements]
labels = [element["label"] for element in elements]
classnames = [element["name"] for element in elements]

prep = [tpt.preprocess(image) for image in images]

trans = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
)

tpt.prompt_learner.prepare_prompts(classnames)

# setup automatic mixed-precision (Amp) loss scaling
scaler = torch.cuda.amp.GradScaler(init_scale=1000)

print("=> Using native Torch AMP. Training in mixed precision.")


LR = 0.005
trainable_param = tpt.prompt_learner.parameters()
optimizer = torch.optim.AdamW(trainable_param, LR)

for i, label in enumerate(labels):

    augs = [trans(prep[i]) for _ in range(4)]
    prep_imgs = torch.tensor(np.stack(augs)).cuda()

    # prep_img = prep[i].unsqueeze(0).cuda()

    for name, param in tpt.named_parameters():
        if "prompt_learner" in name:
            print("LEARNABLE: ", name)

    tpt.eval()

    with torch.no_grad():
        tpt.reset()

    test_time_tuning(tpt, prep_imgs, optimizer)

    with torch.no_grad():
        out = tpt(prep_imgs[0].unsqueeze(0))
    out_id = out.argmax(1).item()

    print(f"Predicted: {classnames[out_id]}\nTarget: {classnames[i]}")
    plt.imshow(out.cpu().detach().numpy(), cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.show()


for i, label in enumerate(labels):
    print(f"Image {i} belongs to class {label}")

# Plot the scores


breakpoint()
