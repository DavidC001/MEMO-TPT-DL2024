import sys

sys.path.append(".")

import torch
from torchvision import transforms
from torch import optim
from torchvision.transforms.v2 import AugMix
import numpy as np
from EasyTPT.tpt_classnames.imagnet_prompts import imagenet_classes
from copy import deepcopy

from dataloaders.dataloader import get_dataloaders, get_classes_names
from EasyTPT.utils import get_datasets

from EasyTPT.models import EasyTPT
from EasyTPT.main import avg_entropy

def TPT_get_model(device, base_prompt="A bad photo of a [CLS].", arch="RN50", splt_ctx= True):
    model = EasyTPT(
        base_prompt=base_prompt,
        arch=arch,
        splt_ctx=splt_ctx,
        classnames=imagenet_classes,
        device=device
    )
        

    return model
def TPT(device, naug=64):
    # prepare TPT
    tpt = TPT_get_model(device=device)

    if not torch.cuda.is_available():
        print("Using CPU this is no bueno")
    else:
        print("Using GPU, brace yourself!")

    datasetRoot = "datasets"
    imageNetA, _, _, _, imageNetV2, _, _, _ = get_datasets(datasetRoot, naug)

    return tpt, imageNetA, imageNetV2

def TPT_inference(tpt:EasyTPT, images, device, temp=1):
    outputs = tpt(images)

    outputs = outputs / temp
    outputs = torch.nn.functional.log_softmax(outputs, dim=1)
    
    return outputs

from memo.utils.adapt_helpers import memo_adapt_single, memo_test_single
from memo.utils.adapt_helpers import memo_transforms
from memo.utils.train_helpers import memo_build_model
def memo(device):
    # prepare MEMO
    memo = memo_build_model(model_name="RN50", device=device, prior_strength=0.94)

    imageNet_A, imageNet_V2 = get_dataloaders('datasets', memo_transforms)

    return memo, imageNet_A, imageNet_V2

def memo_inference(memo, image, device, naug=8, temp=1):
    augmenter = AugMix()

    inputs = [image] + [augmenter(image) for _ in range(naug)]
    inputs = torch.stack(inputs).to(device=device)

    outputs = memo(inputs)

    outputs = outputs / temp
    outputs = torch.nn.functional.log_softmax(outputs, dim=1)

    return outputs

def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

def entropy(logits):
    return -(torch.exp(logits) * logits).sum(dim=-1)

def loss(TPT_outs, memo_outs):
    # calculate the average distribution of the logits, then use entropy to calculate the loss

    # bring the outputs to the same device
    TPT_outs = TPT_outs.to(memo_outs.device)

    #average TPT outputs
    avg_TPT_outs = torch.logsumexp(TPT_outs, dim=0) - torch.log(torch.tensor(TPT_outs.shape[0]))
    min_real = torch.finfo(avg_TPT_outs.dtype).min
    avg_TPT_outs = torch.clamp(avg_TPT_outs, min=min_real)
    #average MEMO outputs
    avg_memo_outs = torch.logsumexp(memo_outs, dim=0) - torch.log(torch.tensor(memo_outs.shape[0]))
    min_real = torch.finfo(avg_memo_outs.dtype).min
    avg_memo_outs = torch.clamp(avg_memo_outs, min=min_real)

    scale = 0.5
    with torch.no_grad():
        memo_ent = entropy(avg_memo_outs)
        TPT_ent = entropy(avg_TPT_outs)
        scale = TPT_ent / (TPT_ent + memo_ent)
    
    #print(f"\t\tscale TPT: {1-scale} - TPT entropy: {TPT_ent}\n\t\tscale MEMO: {scale} - MEMO entropy: {memo_ent}")

    # calculate average logits
    avg_logits = (1-scale) * avg_TPT_outs + scale * avg_memo_outs

    # calculate entropy
    # breakpoint()
    return entropy(avg_logits)

def loss2(TPT_outs, memo_outs):
    # calculate the average distribution of the logits, then use entropy to calculate the loss
    
    #bring the outputs to the same device
    TPT_outs = TPT_outs.to(memo_outs.device)

    #average TPT outputs
    avg_TPT_outs = torch.logsumexp(TPT_outs, dim=0) - torch.log(torch.tensor(TPT_outs.shape[0]))
    min_real = torch.finfo(avg_TPT_outs.dtype).min
    avg_TPT_outs = torch.clamp(avg_TPT_outs, min=min_real)
    TPT_ent = entropy(avg_TPT_outs)
    #average MEMO outputs
    avg_memo_outs = torch.logsumexp(memo_outs, dim=0) - torch.log(torch.tensor(memo_outs.shape[0]))
    min_real = torch.finfo(avg_memo_outs.dtype).min
    avg_memo_outs = torch.clamp(avg_memo_outs, min=min_real)
    memo_ent = entropy(avg_memo_outs)

    return TPT_ent + memo_ent


def printOutput(output, top=5, pre=""):
    classnames = get_classes_names()
    #print top 5 classes
    for i in range(len(output)):
        print(f"{pre}augmentation {i}:")
        _, indices = torch.topk(output[i], top)
        for idx in indices:
            print(f"{pre}\tcLass {idx.item()}: {classnames[idx.item()]}, confidence: {output[i][idx].item()}")


def test(tpt_model:EasyTPT, memo_model, tpt_data, memo_data, device, niter=1):
    classnames = get_classes_names()
    correct = 0
    correctTPT = 0
    correctMEMO = 0
    cnt = 0

    initOptimMEMO = optim.AdamW(memo_model.parameters(), lr= 0.00025, weight_decay=0)

    TPT_temp = 1
    MEMO_temp = 1

    #shuffle the data
    indx = np.random.permutation(range(len(tpt_data)))

    for i in indx:
        cnt += 1
        tpt_model.reset()
        memo = deepcopy(memo_model)
        optimizerMEMO = deepcopy(initOptimMEMO)
        #breakpoint()  

        img_TPT = tpt_data[i]["img"]
        img_MEMO = memo_data[i]["img"].to(device)
        
        label = int(tpt_data[i]["label"])
        label2 = int(memo_data[i]["label"])
        assert label == label2 #check if the labels are the same
        name = tpt_data[i]["name"]

        print (f"Testing on {i} - name: {name} - label: {label}")

        output_base_TPT, _ = tpt_model.predict(img_TPT, ttt_steps=niter)
        if output_base_TPT == label: 
            correctTPT += 1
        print(f"TPT accuracy: {correctTPT/cnt} - tested: {cnt} / {len(tpt_data)}")

        memo_adapt_single(memo, img_MEMO, optimizerMEMO, marginal_entropy, niter, 8, device)
        correctMEMO += memo_test_single(memo, img_MEMO, label, device)[0]
        print(f"MEMO accuracy: {correctMEMO/cnt} - tested: {cnt} / {len(tpt_data)}")

        tpt_model.reset()
        memo = deepcopy(memo_model)
        optimizerMEMO = deepcopy(initOptimMEMO)

        #print ("Ensemble output:")
        for i in range(niter):
            #print(f"\tIteration {i} / {niter}")
            TPT_outs = TPT_inference(tpt_model, img_TPT, "cuda", temp=TPT_temp)
            #print(f"\t\tTPT output:")
            #printOutput(torch.exp(TPT_outs), pre="\t\t\t")
            MEMO_outs = memo_inference(memo, img_MEMO, "cuda", temp=MEMO_temp)
            #print(f"\t\tMEMO output:")
            #printOutput(torch.exp(MEMO_outs), pre="\t\t\t")

            tpt_model.optimizer.zero_grad()
            optimizerMEMO.zero_grad()

            loss_val = loss(TPT_outs, MEMO_outs)
            loss_val.backward()

            tpt_model.optimizer.step()
            optimizerMEMO.step()



        with torch.no_grad():
            TPT_out = torch.log_softmax(tpt_model(img_TPT[0].to(device)) / TPT_temp, dim=1)
            TPT_ent = entropy(TPT_out[0])
            #TPT_prob = torch.exp(TPT_out)
            #print(f"TPT output:")
            #printOutput(TPT_prob, pre="\t")

            MEMO_out = torch.log_softmax(memo(img_MEMO.unsqueeze(0).to(device)) / MEMO_temp, dim=1)
            MEMO_ent = entropy(MEMO_out[0])
            #MEMO_prob = torch.exp(MEMO_out)
            #print(f"MEMO output:")
            #printOutput(MEMO_prob, pre="\t")

            TPT_out = TPT_out.to(MEMO_out.device) #bring the outputs to the same device

            scale = TPT_ent / (TPT_ent + MEMO_ent)

            #ensemble
            out = TPT_out * (1-scale) + MEMO_out * scale
            #to probability
            out = torch.exp(out)
            #print(f"Ensemble output:")
            #printOutput(out, pre="\t")

            #breakpoint()
            #get max as prediction
            _, predicted = out.max(1)

            if predicted.item() == label:
                correct += 1
            

            print(f"Ensemble accuracy: {correct/cnt} - predicted: {classnames[predicted.item()]} - label: {name} - tested: {cnt} / {len(tpt_data)}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #set the seed
    torch.manual_seed(0)
    np.random.seed(0)

    tpt_model, tpt_dataA, tpt_dataV2 = TPT("cuda")
    memo_model, memo_dataA, memo_dataV2 = memo("cuda")
    
    print("Testing on ImageNet-A")
    test(tpt_model, memo_model, tpt_dataA, memo_dataA, device)
    print("\nTesting on ImageNet-V2")
    test(tpt_model, memo_model, tpt_dataV2, memo_dataV2, device)


if __name__ == "__main__":
    main()
