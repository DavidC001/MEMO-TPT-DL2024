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
from dataloaders.dataloader import get_classes_names
from models import EasyTPT

from clip import tokenize

torch.autograd.set_detect_anomaly(True)


def select_confident_samples(logits, top):

    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[
        : int(batch_entropy.size()[0] * top)
    ]
    return logits[idx], idx


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


def test_time_tuning(model, inputs, optimizer, ttt_steps=4):

    model.eval()
    selected_idx = None
    for j in range(ttt_steps):

        output = model(inputs)
        if selected_idx is not None:
            output = output[selected_idx]
        else:
            output, selected_idx = select_confident_samples(output, 0.10)

        loss = avg_entropy(output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return


def tpt_eval(model, imgs, optimizer, ttt_steps, optim_state):

    og_img = imgs[0].unsqueeze(0)
    prep_imgs = torch.stack(imgs).cuda()

    model.eval()

    with torch.no_grad():
        model.reset()

    optimizer.load_state_dict(optim_state)

    test_time_tuning(model, prep_imgs, optimizer, ttt_steps=ttt_steps)
    with torch.no_grad():
        out = model(og_img.cuda())
    out_id = out.argmax(1).item()
    return out_id


def clip_eval(model, img_prep):
    tkn_prompts = tokenize(model.prompt_learner.txt_prompts)

    with torch.no_grad():
        image_feat = model.clip.encode_image(img_prep.unsqueeze(0).cuda())
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        txt_feat = model.clip.encode_text(tkn_prompts.cuda())
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    logit_scale = model.clip.logit_scale.exp()
    logits = logit_scale * image_feat @ txt_feat.t()
    clip_id = logits.argmax(1).item()
    return clip_id


device = "cuda:0"

ARCH = "RN50"
BASE_PROMPT = "a photo of a [CLS]"
SPLT_CTX = True

tpt = EasyTPT(device, base_prompt=BASE_PROMPT, arch=ARCH, splt_ctx=SPLT_CTX)


for name, param in tpt.named_parameters():
    param.requires_grad_(False)


if not torch.cuda.is_available():
    print("Using CPU this is no bueno")
else:
    print("Switching to GPU, brace yourself!")
    torch.cuda.set_device(device)
    tpt = tpt.cuda(device)

transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224))])
ima_root = "datasets/imagenet-a"
dataset = ImageNetA(ima_root, transform=transform)

# imv_root = "datasets/imagenetv2-matched-frequency-format-val"
# dataset = ImageNetV2(imv_root, transform=transform)

my_classes = dataset.getClassesNames()
all_classnames = get_classes_names()

# some fuckery to use the original TPT prompts
from tpt_classnames.imagnet_prompts import imagenet_classes
from tpt_classnames.imagenet_variants import imagenet_a_mask

label_mask = eval("imagenet_a_mask")
classnames = [imagenet_classes[i] for i in label_mask]

a_classes = [(i, name) for i, name in enumerate(all_classnames) if name in my_classes]
a_classnames = [name for name in all_classnames if name in my_classes]
a_classnames = classnames

from utils import augmix

base_trans = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
)


tpt.prompt_learner.prepare_prompts(a_classnames)


LR = 0.005
trainable_param = tpt.prompt_learner.parameters()
optimizer = torch.optim.AdamW(trainable_param, LR)
optim_state = deepcopy(optimizer.state_dict())

tpt_correct = 0
clip_correct = 0
cnt = 0


idxs = list(range(len(dataset)))

TTT_STEPS = 1
AUGMIX = False
NAUG = 63

EVAL_CLIP = False

for _ in range(len(idxs)):

    idx = choice(idxs)
    idxs.remove(idx)
    data = dataset[idx]
    img = data["img"]
    label = data["label"]
    name = data["name"]

    img_prep = tpt.preprocess(img)

    if AUGMIX:
        augs = [img_prep] + [augmix(img, tpt.preprocess) for _ in range(NAUG)]
    else:
        augs = [img_prep] + [base_trans(img_prep) for _ in range(NAUG)]

    out_id = tpt_eval(tpt, augs, optimizer, TTT_STEPS, optim_state)

    og_label, og_classname = a_classes[out_id]

    if og_label == int(label):
        tpt_correct += 1
    cnt += 1

    tpt_acc = tpt_correct / (cnt)

    ################ CLIP ############################
    if EVAL_CLIP:
        clip_id = clip_eval(tpt, img_prep)

        clip_label, clip_classname = a_classes[clip_id]

        if clip_label == int(label):
            clip_correct += 1

        clip_acc = clip_correct / (cnt)
    ###################################################

    print(f"TPT Accuracy: {round(tpt_acc,3)}")
    if EVAL_CLIP:
        print(f"CLIP Accuracy: {round(clip_acc,3)}")
    print(f"GT: \t{name}\nTPT: \t{og_classname}")
    if EVAL_CLIP:
        print(f"CLIP: \t{clip_classname}")
    print(f"after {cnt} samples\n")
breakpoint()
