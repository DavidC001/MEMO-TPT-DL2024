import sys

sys.path.append(".")

import torch
from torchvision import transforms
from torch import optim
from torchvision.transforms.v2 import AugMix
import numpy as np

from dataloaders.dataloader import get_dataloaders, get_classes_names


from EasyTPT.main import select_confident_samples
from EasyTPT.models import EasyTPT


def TPT_tpt_get_transforms():
    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((224, 224))]
    )
    return transform


def TPT_get_model(
    device,
    base_prompt="This is a photo of a [CLS], ImageNet",
    arch="RN50",
    splt_ctx=True,
):
    model = EasyTPT(
        base_prompt=base_prompt, arch=arch, splt_ctx=splt_ctx, device=device
    )
    for name, param in model.named_parameters():
        param.requires_grad_(False)

    return model


def TPT_get_optimizer(model, lr=0.005):
    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, lr)
    return optimizer


def TPT(device):
    # prepare TPT
    tpt = TPT_get_model(device=device)

    if not torch.cuda.is_available():
        print("Using CPU this is no bueno")
    else:
        print("Using GPU, brace yourself!")

    datasetRoot = "datasets"
    imageNetA, imageNetV2 = get_dataloaders(
        datasetRoot, transform=TPT_tpt_get_transforms()
    )

    classnames = get_classes_names()

    tpt.prompt_learner.prepare_prompts(classnames)

    return tpt, imageNetA, imageNetV2


def TPT_inference(tpt: EasyTPT, image, device, naug=31):
    img_prep = tpt.preprocess(image)

    augmix = AugMix()
    inputs = [img_prep] + [tpt.preprocess(augmix(image)) for _ in range(naug)]
    inputs = torch.stack(inputs).to(device=device)

    outputs = tpt(inputs)
    outputs, _ = select_confident_samples(outputs, 0.10)

    return outputs


from memo.utils.adapt_helpers import te_transforms
from memo.utils.adapt_helpers import adapt_single, test_single
from memo.utils.train_helpers import build_model


def memo(device):
    # prepare MEMO
    memo = build_model(model_name="RN50", device=device, prior_strength=0.94)

    imageNet_A, imageNet_V2 = get_dataloaders("datasets", te_transforms)

    return memo, imageNet_A, imageNet_V2


def memo_inference(memo, image, device, naug=8):
    augmenter = AugMix()

    inputs = [image] + [augmenter(image) for _ in range(naug)]
    inputs = torch.stack(inputs).to(device=device)

    outputs = memo(inputs)

    return outputs


def loss(TPT_outs, memo_outs):
    # calculate the average distribution of the logits, then use entropy to calculate the loss

    # bring the outputs to the same device
    TPT_outs = TPT_outs.to(memo_outs.device)

    # calculate logits
    TPT_logits = TPT_outs - TPT_outs.logsumexp(dim=-1, keepdim=True)
    memo_logits = memo_outs - memo_outs.logsumexp(dim=-1, keepdim=True)
    all_logits = torch.cat([TPT_logits, memo_logits], dim=0)

    # calculate average logits
    avg_logits = all_logits.logsumexp(dim=0) - np.log(all_logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)

    # calculate entropy
    entropy = -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

    return entropy


def test(tpt_model: EasyTPT, memo_model, tpt_data, memo_data, device, niter=1):
    classnames = get_classes_names()
    correct = 0
    cnt = 0
    memo_save = memo_model.state_dict()

    optimizerTPT = TPT_get_optimizer(tpt_model)
    initOptimTPT = optimizerTPT.state_dict()
    optimizerMEMO = optim.AdamW(memo_model.parameters(), lr=0.01, weight_decay=0.01)
    initOptimMEMO = optimizerMEMO.state_dict()

    for i in range(len(tpt_data)):
        tpt_model.reset()
        optimizerTPT.load_state_dict(initOptimTPT)
        memo_model.load_state_dict(memo_save)
        optimizerMEMO.load_state_dict(initOptimMEMO)

        data_TPT = tpt_data[i]
        data_MEMO = memo_data[i]

        img_TPT = data_TPT["img"]
        img_MEMO = data_MEMO["img"]

        label = int(data_TPT["label"])
        name = data_TPT["name"]

        for _ in range(niter):
            TPT_outs = TPT_inference(tpt_model, img_TPT, "cuda")
            MEMO_outs = memo_inference(memo_model, img_MEMO, "cuda")

            loss_val = loss(TPT_outs, MEMO_outs)
            loss_val.backward()

            optimizerTPT.step()
            optimizerMEMO.step()

        with torch.no_grad():
            img_prep = tpt_model.preprocess(img_TPT).unsqueeze(0).to("cuda")
            TPT_out = tpt_model(img_prep)
            MEMO_out = memo_model(img_MEMO.unsqueeze(0).to("cuda"))

            # bring the outputs to the same device
            TPT_out = TPT_out.to(MEMO_out.device)
            out = TPT_out + MEMO_out
            # get max as prediction
            _, predicted = out.max(1)

            if predicted.item() == label:
                correct += 1

            cnt += 1

            print(
                f"\tAccuracy: {correct/cnt} - predicted: {classnames[predicted.item()]} - label: {name} - tested: {cnt} / {len(tpt_data)}"
            )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tpt_model, tpt_dataA, tpt_dataV2 = TPT("cuda")
    memo_model, memo_dataA, memo_dataV2 = memo("cuda")

    print("Testing on ImageNet-A")
    test(tpt_model, memo_model, tpt_dataA, memo_dataA, device)
    print("\nTesting on ImageNet-V2")
    test(tpt_model, memo_model, tpt_dataV2, memo_dataV2, device)


if __name__ == "__main__":
    main()
