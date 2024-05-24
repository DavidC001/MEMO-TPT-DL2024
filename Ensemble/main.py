import sys
sys.path.append(".")

import torch
from torch import nn
from torchvision import transforms
from torch import optim
from torchvision.transforms.v2 import AugMix
import numpy as np
from EasyTPT.tpt_classnames.imagnet_prompts import imagenet_classes
from copy import deepcopy

from dataloaders.dataloader import get_classes_names
from EasyTPT.utils import tpt_get_datasets
from EasyTPT.models import EasyTPT

from Ensemble.models import Ensemble

def TPT(device="cuda", naug=30, arch="RN50", A=True, ttt_steps=1, align_steps=0):
    # prepare TPT
    if not torch.cuda.is_available():
        print("Using CPU this is no bueno")
    else:
        print("Using GPU, brace yourself!")

    datasetRoot = "datasets"
    imageNetA, _, imageNetACustomNames, imageNetAMap, imageNetV2, _, imageNetV2CustomNames, imageNetV2Map = tpt_get_datasets(datasetRoot, augs=naug, all_classes=False)
    
    if A:
        dataset = imageNetA
        classnames = imageNetACustomNames
        mapping = imageNetAMap
    else:
        dataset = imageNetV2
        classnames = imageNetV2CustomNames
        mapping = imageNetV2Map
    
    tpt = EasyTPT(
        base_prompt="A bad photo of a [CLS]",
        arch=arch,
        classnames=classnames,
        device=device,
        ttt_steps=ttt_steps,
        align_steps=align_steps
    )
    
    return tpt, dataset, mapping


from memo.models import memo_get_datasets, EasyMemo
from torchvision.models import resnet50, ResNet50_Weights

def memo(device="cuda", prior_strength=0.94, naug=30, A=True, drop=0, ttt_steps=1):
    # prepare MEMO
    imageNet_A, imageNet_V2 = memo_get_datasets(augmentation=('cut' if drop==0 else 'identity'), augs=naug)
    dataset = imageNet_A if A else imageNet_V2

    mapping = list(dataset.classnames.keys())
    for i,id in enumerate(mapping):
        mapping[i] = int(id)
    
    rn50 = resnet50(weights=ResNet50_Weights.DEFAULT)
    rn50.layer4.add_module('dropout', nn.Dropout(drop))

    memo = EasyMemo(
        rn50, 
        device=device, 
        classes_mask=mapping, 
        prior_strength=prior_strength,
        niter=ttt_steps,
        ensemble=(drop>0)
    )
    
    return memo, dataset



def test(tpt_model:EasyTPT, memo_model, tpt_data, mapping, memo_data, 
         device="cuda", niter=1, top=0.1,
         no_backwards=False, testSingleModels=False):
    correct = 0
    correct_no_back = 0
    correctSingle = [0, 0]
    cnt = 0

    class_names = get_classes_names()
    models_names = ["TPT", "MEMO"]

    TPT_temp = 1.55
    MEMO_temp = 0.7
    temps = [TPT_temp, MEMO_temp]

    #shuffle the data
    indx = np.random.permutation(range(len(tpt_data)))

    model = Ensemble(models=[tpt_model, memo_model], temps=temps, 
                     device=device, test_single_models=testSingleModels, 
                     no_backwards=no_backwards)
    print("Ensemble model created starting TTA, samples:",len(indx))
    for i in indx:
        cnt += 1 
        img_TPT = tpt_data[i]["img"]
        img_MEMO = memo_data[i]["img"]
        data = [img_TPT, img_MEMO]
        
        label = int(tpt_data[i]["label"])
        label2 = int(memo_data[i]["label"])
        assert label == label2 #check if the labels are the same
        name = tpt_data[i]["name"]

        print (f"Testing on {i} - name: {name} - label: {label}")

        models_out, pred_no_back, prediction = model(data, niter=niter, top=top)
        models_out = [int(mapping[model_out]) for model_out in models_out]
        prediction = int(mapping[prediction])
        
        if testSingleModels:
            for i, model_out in enumerate(models_out):
                if label == model_out:
                    correctSingle[i] += 1
                
                print(f"\t{models_names[i]} model accuracy: {correctSingle[i]}/{cnt} - predicted class {model_out}: {class_names[model_out]} - tested: {cnt} / {len(tpt_data)}")

        if no_backwards:
            pred_no_back = int(mapping[pred_no_back])
            if label == pred_no_back:
                correct_no_back += 1
            print(f"\tSimple Ens accuracy: {correct_no_back}/{cnt} - predicted class {pred_no_back}: {class_names[pred_no_back]} - tested: {cnt} / {len(tpt_data)}")

        if label == prediction:
            correct += 1
            
        print(f"\tEnsemble accuracy: {correct}/{cnt} - predicted class {prediction}: {class_names[prediction]} - tested: {cnt} / {len(tpt_data)}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    imageNetA = True
    naug = 64
    top = 0.1
    niter = 1
    testSingleModels = False
    no_backwards = True

    #set the seed
    torch.manual_seed(0)
    np.random.seed(0)

    #TPT
    TPT_device = "cuda" if torch.cuda.is_available() else "cpu"
    TPT_align_steps = 0
    TPT_arch = "RN50"
    TPT_iters = 1
    tpt_model, tpt_data, mapping = TPT(device=TPT_device, naug=naug, arch=TPT_arch, A=imageNetA, ttt_steps=TPT_iters, align_steps=TPT_align_steps)
    
    #MEMO
    MEMO_device = "cuda" if torch.cuda.is_available() else "cpu"
    MEMO_drop = 0
    MEMO_prior_strength = 0.94
    MEMO_iters = 1
    memo_model, memo_data = memo(device=MEMO_device, naug=naug, A=imageNetA, ttt_steps=MEMO_iters, drop=MEMO_drop, prior_strength=MEMO_prior_strength)

    if (imageNetA):
        print("Testing on ImageNet-A")
    else:
        print("Testing on ImageNet-V2")

    torch.autograd.set_detect_anomaly(True)
    test(tpt_model, memo_model, tpt_data, mapping, memo_data, device, niter, top, no_backwards, testSingleModels)

if __name__ == "__main__":
    main()