import sys

sys.path.append(".")


import torch
import numpy as np

from pprint import pprint
from clip import tokenize

from dataloaders.imageNetA import ImageNetA

from EasyTPT.utils import tpt_get_transforms, tpt_get_datasets
from EasyTPT.models import EasyTPT
from EasyTPT.setup import get_args
from EasyTPT.tpt_classnames.imagnet_prompts import imagenet_classes
from EasyTPT.tpt_classnames.imagenet_variants import imagenet_a_mask

torch.autograd.set_detect_anomaly(True)


def tpt_clip_eval(model, img_prep):
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
    EMB_ALIGN = args["emb_align"]

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
        augs=AUGS,
        lr=LR,
        # emb_align=EMB_ALIGN,
        emb_align=False,
    )

    tpt_correct = 0
    clip_correct = 0
    cnt = 0

    idxs = [i for i in range(len(imageNet_A))]

    seed = 0
    np.random.seed(seed)
    np.random.shuffle(idxs)
    # for i, data in enumerate(imageNet_A):
    for i in idxs:
        data = imageNet_A[i]
        label = data["label"]
        imgs = data["img"]
        name = data["name"]

        with torch.no_grad():
            tpt.reset()

        optimizer = EasyTPT.get_optimizer(tpt)

        for _ in range(TTT_STEPS):

            out = tpt(imgs)
            loss = tpt_avg_entropy(out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            out = tpt(imgs[0])
            out_id = out.argmax(1).item()
            tpt_predicted = classnames[out_id]
            # if True:
            #     out_id, tpt_predicted = tpt.predict(imgs, ttt_steps=TTT_STEPS)
            if id_mapping[out_id] == label:
                print(":)")
                tpt_correct += 1
            else:
                print(":(")
            cnt += 1

            tpt_acc = tpt_correct / (cnt)

        ################ CLIP ############################
        if EVAL_CLIP:
            clip_id = tpt_clip_eval(tpt, imgs)
            clip_predicted = classnames[clip_id]
            if id_mapping[clip_id] == label:
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
