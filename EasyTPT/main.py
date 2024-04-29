import sys

sys.path.append(".")


import torch


from pprint import pprint
from clip import tokenize

from dataloaders.imageNetA import ImageNetA

from EasyTPT.utils import get_transforms
from EasyTPT.models import EasyTPT
from EasyTPT.setup import get_args
from EasyTPT.tpt_classnames.imagnet_prompts import imagenet_classes
from EasyTPT.tpt_classnames.imagenet_variants import imagenet_a_mask

torch.autograd.set_detect_anomaly(True)


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
    AUGS = args["augs"]
    TTT_STEPS = args["tts"]
    AUGMIX = args["augmix"]
    ######## DATALOADER #############################################

    ima_root = "datasets/imagenet-a"
    datasetRoot = "datasets"
    imageNet_A = ImageNetA(ima_root, transform=get_transforms(augs=AUGS))
    # breakpoint()
    # val_dataset = DatasetWrapper(ima_root, transform=data_transform)

    print("number of test samples: {}".format(len(imageNet_A)))
    val_loader = torch.utils.data.DataLoader(
        imageNet_A,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
    )

    ##############################################################################
    # makes sure the class idx has the right correspondece
    # to the class label
    label_mask = eval("imagenet_a_mask")
    classnames = [imagenet_classes[i] for i in label_mask]

    ima_id_mapping = list(imageNet_A.classnames.keys())
    ima_names = list(imageNet_A.classnames.values())

    ima_custom_names = [imagenet_classes[int(i)] for i in ima_id_mapping]
    # breakpoint()
    # imv2_id_mapping = list(imageNetV2.classnames.keys())
    # imv2_names = list(imageNetV2.classnames.values())

    classnames = ima_custom_names
    id_mapping = ima_id_mapping

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
    )

    tpt_correct = 0
    clip_correct = 0
    cnt = 0

    EVAL_CLIP = args["clip"]

    for i, data in enumerate(val_loader):

        label = data["label"][0]
        imgs = data["img"]
        name = data["name"][0]

        with torch.no_grad():
            tpt.reset()

        out = tpt(imgs)

        out_id = out.argmax(1).item()

        with torch.no_grad():
            tpt_predicted = classnames[out_id]

            if id_mapping[out_id] == label:
                print(":)")
                tpt_correct += 1
            else:
                print(":(")
            cnt += 1

            tpt_acc = tpt_correct / (cnt)

        ################ CLIP ############################
        if EVAL_CLIP:
            clip_id = clip_eval(tpt, imgs)
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
