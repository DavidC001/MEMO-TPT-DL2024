import sys

sys.path.append(".")

import os
import random
import math
import torch

import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from tqdm import tqdm
from random import choice
from PIL import Image
from pprint import pprint
from clip import tokenize

from torchvision.transforms import InterpolationMode
from torchvision import transforms
from torch.utils.data import DataLoader

from dataloaders.dataloader import get_dataloaders
from dataloaders.imageNetA import ImageNetA
from dataloaders.imageNetV2 import ImageNetV2
from dataloaders.dataloader import get_classes_names

from EasyTPT.utils import DatasetWrapper, EasyAgumenter
from EasyTPT.models import EasyTPT
from EasyTPT.setup import get_args
from EasyTPT.tpt_classnames.imagnet_prompts import imagenet_classes
from EasyTPT.tpt_classnames.imagenet_variants import imagenet_a_mask

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


def tpt_inference(model, imgs, optimizer, ttt_steps, optim_state):

    image = imgs[0].cuda()

    images = torch.cat(imgs, dim=0).cuda()

    model.eval()

    with torch.no_grad():
        model.reset()

    optimizer.load_state_dict(optim_state)

    test_time_tuning(model, images, optimizer, ttt_steps=ttt_steps)
    with torch.no_grad():
        out = model(image.cuda())
        out_id = out.argmax(1).item()
    return out_id


def clip_eval(model, img_prep):
    tkn_prompts = tokenize(model.prompt_learner.txt_prompts)

    with torch.no_grad():
        image_feat = model.clip.encode_image(img_prep[0].cuda())
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        txt_feat = model.clip.encode_text(tkn_prompts.cuda())
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    logit_scale = model.clip.logit_scale.exp()
    logits = logit_scale * image_feat @ txt_feat.t()
    clip_id = logits.argmax(1).item()
    return clip_id


def main():
    args = get_args()
    pprint(args)

    device = "cuda:0"

    ARCH = args["arch"]
    BASE_PROMPT = args["base_prompt"]
    SPLT_CTX = not args["single_context"]

    tpt = EasyTPT(device, base_prompt=BASE_PROMPT, arch=ARCH, splt_ctx=SPLT_CTX)

    # freeze the model
    for name, param in tpt.named_parameters():
        param.requires_grad_(False)

    # switch to GPU if available
    if not torch.cuda.is_available():
        print("Using CPU this is no bueno")
    else:
        print("Switching to GPU, brace yourself!")
        torch.cuda.set_device(device)
        tpt = tpt.cuda(device)

    ######## DATALOADER #############################################
    base_transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
    )

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    data_transform = EasyAgumenter(
        base_transform,
        preprocess,
        n_views=63,
    )
    ima_root = "datasets/imagenet-a"

    # datasetRoot = "datasets"
    # imageNetA, imageNetV2 = get_dataloaders(datasetRoot, transform=data_transform())
    # breakpoint()
    val_dataset = DatasetWrapper(ima_root, transform=data_transform)

    print("number of test samples: {}".format(len(val_dataset)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
    )

    ##############################################################################

    # some fuckery to use the original TPT prompts

    label_mask = eval("imagenet_a_mask")
    classnames = [imagenet_classes[i] for i in label_mask]

    # Initialize EasyPromptLearner
    tpt.prompt_learner.prepare_prompts(classnames)

    LR = 0.005
    trainable_param = tpt.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, LR)
    optim_state = deepcopy(optimizer.state_dict())

    tpt_correct = 0
    clip_correct = 0
    cnt = 0

    TTT_STEPS = args["tts"]
    AUGMIX = args["augmix"]
    NAUG = 63

    EVAL_CLIP = args["clip"]

    for i, (imgs, target) in enumerate(val_loader):

        label = target[0]
        path = target[1]
        name = classnames[int(label)]

        out_id = tpt_inference(tpt, imgs, optimizer, TTT_STEPS, optim_state)
        with torch.no_grad():
            tpt_predicted = classnames[out_id]

            if out_id == int(label):
                tpt_correct += 1
            cnt += 1

            tpt_acc = tpt_correct / (cnt)

        ################ CLIP ############################
        if EVAL_CLIP:
            clip_id = clip_eval(tpt, imgs)
            clip_predicted = classnames[clip_id]
            if clip_id == int(label):
                clip_correct += 1

            clip_acc = clip_correct / (cnt)
        ###################################################

        print(f"TPT Accuracy: {round(tpt_acc,3)}")
        if EVAL_CLIP:
            print(f"CLIP Accuracy: {round(clip_acc,3)}")
        print(f"GT: \t{name}\nTPT: \t{tpt_predicted}")
        if EVAL_CLIP:
            print(f"CLIP: \t{clip_predicted}")
        print(f"after {cnt} samples\n")
    breakpoint()


if __name__ == "__main__":
    main()
