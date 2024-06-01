import sys

sys.path.append(".")


import torch
import numpy as np
import random

from pprint import pprint
from clip import tokenize

from dataloaders.imageNetA import ImageNetA

from EasyTPT.utils import tpt_get_transforms, tpt_get_datasets
from EasyTPT.models import EasyTPT
from EasyTPT.setup import get_args
from EasyTPT.tpt_classnames.imagnet_prompts import imagenet_classes
from EasyTPT.tpt_classnames.imagenet_variants import imagenet_a_mask

torch.autograd.set_detect_anomaly(True)


def tpt_avg_entropy(outputs):

    logits = outputs - outputs.logsumexp(
        dim=-1, keepdim=True
    )  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(
        logits.shape[0]
    )  # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def main():
    args = get_args()
    pprint(args)

    device = "cuda:0"

    ARCH = args["arch"]
    BASE_PROMPT = args["base_prompt"]
    SPLT_CTX = not args["single_context"]
    AUGS = args["augs"]
    TTT_STEPS = args["tts"]
    AUGMIX = args["augmix"]
    EVAL_CLIP = args["clip"]
    ALIGN_STEPS = args["align_steps"]
    ENSEMBLE = args["ensemble"]

    data_root = "datasets"

    (
        imageNet_A,
        ima_names,
        ima_custom_names,
        ima_label_mapping,
        imageNet_V2,
        imv2_names,
        imv2_custom_names,
        imv2_label_mapping,
    ) = tpt_get_datasets(data_root, augmix=AUGMIX, augs=AUGS, all_classes=False)

    print("number of test samples: {}".format(len(imageNet_A)))

    classnames = ima_custom_names
    id_mapping = ima_label_mapping

    LR = 0.005

    tpt = EasyTPT(
        device,
        base_prompt=BASE_PROMPT,
        arch=ARCH,
        splt_ctx=SPLT_CTX,
        classnames=classnames,
        ttt_steps=TTT_STEPS,
        lr=LR,
        align_steps=ALIGN_STEPS,
        ensemble=ENSEMBLE,
        confidence=0.10,
    )

    tpt_correct = 0
    cnt = 0

    idxs = [i for i in range(len(imageNet_A))]

    SEED = 1
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    np.random.shuffle(idxs)
    # for i, data in enumerate(imageNet_A):
    for i in idxs:
        data = imageNet_A[i]
        label = data["label"]
        imgs = data["img"]
        name = data["name"]

        with torch.no_grad():
            tpt.reset()
        out_id = tpt.predict(imgs)
        tpt_predicted = classnames[out_id]

        if int(id_mapping[out_id]) == label:
            print(":D")
            tpt_correct += 1
        else:
            print(":(")
        cnt += 1

        tpt_acc = tpt_correct / (cnt)

        print(f"TPT Accuracy: {round(tpt_acc,3)}")
        print(f"GT: \t{name}\nTPT: \t{tpt_predicted}")
        print(f"after {cnt} samples\n")
    # breakpoint()


if __name__ == "__main__":
    main()
