import sys

sys.path.append(".")


import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import (
    Summary,
    AverageMeter,
    ProgressMeter,
    accuracy,
    load_model_weight,
    set_random_seed,
)
from data.cls_to_names import *
from data.imagenet_variants import (
    thousand_k_to_200,
    imagenet_a_mask,
    imagenet_r_mask,
    imagenet_v_mask,
)

from dataloaders.imageNetA import ImageNetA

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


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


def test_time_tuning(model, inputs, optimizer, scaler, args):

    selected_idx = None
    for j in range(args["tta_steps"]):
        with torch.cuda.amp.autocast():

            output = model(inputs)

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(
                    output, args["selection_p"]
                )

            loss = avg_entropy(output)

        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()

    return


def test_time_single(images, model, model_state, optimizer, optim_state, scaler, args):

    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    # reset model and switch to evaluate mode
    model.eval()

    with torch.no_grad():
        model.reset()

    assert args["gpu"] is not None
    for k in range(len(images)):
        images[k] = images[k].cuda(args["gpu"], non_blocking=True)
    image = images[0]

    images = torch.cat(images, dim=0)

    # reset the tunable prompt to its initial state
    if args["tta_steps"] > 0:
        with torch.no_grad():
            model.reset()
    optimizer.load_state_dict(optim_state)
    test_time_tuning(model, images, optimizer, scaler, args)

    # The actual inference goes here
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model(image)

    return output


def test_time_adapt_eval(
    val_loader, model, model_state, optimizer, optim_state, scaler, args
):

    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    # reset model and switch to evaluate mode
    model.eval()

    with torch.no_grad():
        model.reset()

    for i, (images, target) in enumerate(val_loader):
        print(f"Processing batch {i}...")
        assert args["gpu"] is not None
        for k in range(len(images)):
            images[k] = images[k].cuda(args["gpu"], non_blocking=True)
        image = images[0]

        target = target.cuda(args["gpu"], non_blocking=True)

        images = torch.cat(images, dim=0)

        # reset the tunable prompt to its initial state
        if args["tta_steps"] > 0:
            with torch.no_grad():
                model.reset()
        optimizer.load_state_dict(optim_state)
        test_time_tuning(model, images, optimizer, scaler, args)

        # The actual inference goes here
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(image)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        print(f"top1: {top1.avg}, top5: {top5.avg}")
    return [top1.avg, top5.avg]


if __name__ == "__main__":

    args = {
        "data": "/home/lollo/Downloads/",
        "test_sets": "A",
        "dataset_mode": "test",
        "arch": "RN50",
        "resolution": 224,
        "workers": 4,
        "batch_size": 64,
        "lr": 0.005,
        "print_freq": 200,
        "gpu": 0,
        "tpt": True,
        "selection_p": 0.1,
        "tta_steps": 1,
        "n_ctx": 4,
        "ctx_init": "a_photo_of_a",
        "cocoop": False,
        "load": None,
        "seed": 0,
    }
    set_random_seed(args["seed"])

    # This codebase has only been tested under the single GPU setting
    assert args["gpu"] is not None

    set_random_seed(args["seed"])
    print("Use GPU: {} for training".format(args["gpu"]))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)

    classnames = imagenet_classes

    model = get_coop(
        args["arch"], args["test_sets"], args["gpu"], args["n_ctx"], args["ctx_init"]
    )

    model_state = None
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    print("=> Model created: visual backbone {}".format(args["arch"]))

    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    else:
        assert args["gpu"] is not None
        torch.cuda.set_device(args["gpu"])
        model = model.cuda(args["gpu"])

    # define optimizer
    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args["lr"])
    optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    # This prevents underflow for floating point operations
    # expecially when using lower precision floating points
    # for performance benefits. It's usually used together
    # autocast that wnables mixed precision trainigg
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print("=> Using native Torch AMP. Training in mixed precision.")

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    # temporarly hardcoding imagenet-a
    set_id = "A"
    results = {}

    # TODO: check correctness of this but this should be where
    # the 63 agumented views are created. The batch size corresponds
    # to the number of augmented views -1
    base_transform = transforms.Compose(
        [
            transforms.Resize(args["resolution"], interpolation=BICUBIC),
            transforms.CenterCrop(args["resolution"]),
        ]
    )
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])
    data_transform = AugMixAugmenter(
        base_transform,
        preprocess,
        n_views=args["batch_size"] - 1,
        augmix=len(set_id) > 1,
    )
    # set batch size to 1 again (image is 1 + 63 views)
    batchsize = 1

    print("evaluating: {}".format(set_id))
    # reset the model
    # Reset classnames of custom CLIP model

    # masks classnames to the subset of the dataset. If all 1k classes
    # are used it fills all the memory. Masks are defined in data/imagenet_variants.py
    classnames_all = imagenet_classes
    classnames = []
    label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
    classnames = [classnames_all[i] for i in label_mask]

    # called inside PromptLearner, it tokenizes the promts and saves
    # prefixes and suffixes
    model.reset_classnames(classnames, args["arch"])

    # loads the DataLoader TODO: put our own dataloader here
    val_dataset = build_dataset(
        set_id, data_transform, args["data"], mode=args["dataset_mode"]
    )

    print("number of test samples: {}".format(len(val_dataset)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=args["workers"],
        pin_memory=True,
    )

    # results[set_id] = test_time_adapt_eval(
    #     val_loader, model, model_state, optimizer, optim_state, scaler, args
    # )

    imgs, target = next(iter(val_loader))

    out = test_time_single(
        imgs, model, model_state, optimizer, optim_state, scaler, args
    )

    out_id = out.argmax(1).item()
    target_id = target.item()
    print(f"Predicted: {classnames[out_id]}, Target: {classnames[target_id]}")
    breakpoint()
