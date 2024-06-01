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

def TPT(device="cuda", naug=30, arch="RN50", A=True, ttt_steps=1, align_steps=0, top=0.1):
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
        align_steps=align_steps,
        confidence=top
    )
    
    return tpt, dataset, mapping


from memo.models import memo_get_datasets, EasyMemo
import torchvision.models as models

def memo(device="cuda", prior_strength=0.94, naug=30, A=True, drop=0, ttt_steps=1, model="RN50", top=0.1):
    load_model = {
        "RN50": models.resnet50,
        "RNXT": models.resnext50_32x4d
    }
    models_weights = {
        "RN50": models.ResNet50_Weights.DEFAULT,
        "RNXT": models.ResNeXt50_32X4D_Weights.DEFAULT
    }
    # prepare MEMO
    imageNet_A, imageNet_V2 = memo_get_datasets(augmentation=('cut' if drop==0 else 'identity'), augs=naug)
    dataset = imageNet_A if A else imageNet_V2

    mapping = list(dataset.classnames.keys())
    for i,id in enumerate(mapping):
        mapping[i] = int(id)
    
    model = load_model[model](weights=models_weights[model])
    model.layer4.add_module('dropout', nn.Dropout(drop))

    memo = EasyMemo(
        model, 
        device=device, 
        classes_mask=mapping, 
        prior_strength=prior_strength,
        niter=ttt_steps,
        ensemble=(drop>0),
        top=top
    )
    
    return memo, dataset, mapping


def test(models, datasets, temps, mapping, names,
         device="cuda", niter=1, top=0.1,
         simple_ensemble=False, testSingleModels=False):
    correct = 0
    correct_no_back = 0
    correctSingle = [0]*len(models)
    cnt = 0

    class_names = get_classes_names()

    #shuffle the data
    indx = np.random.permutation(range(len(datasets[0])))

    model = Ensemble(models, temps=temps, 
                     device=device, test_single_models=testSingleModels, 
                     simple_ensemble=simple_ensemble)
    print("Ensemble model created starting TTA, samples:",len(indx))
    for i in indx:
        cnt += 1 
        data = [datasets[j][i]["img"] for j in range(len(datasets))]
        
        labels = [datasets[j][i]["label"] for j in range(len(datasets))]
        #check if the labels are the same
        assert all(x == labels[0] for x in labels), "Labels are not the same"
        label = labels[0]
        name = datasets[0][i]["name"]

        print (f"Testing on {i} - name: {name} - label: {label}")

        models_out, pred_no_back, prediction = model(data, niter=niter, top=top)
        models_out = [int(mapping[model_out]) for model_out in models_out]
        prediction = int(mapping[prediction])
        
        if testSingleModels:
            for i, model_out in enumerate(models_out):
                if label == model_out:
                    correctSingle[i] += 1
                
                print(f"\t{names[i]} model accuracy: {correctSingle[i]}/{cnt} - predicted class {model_out}: {class_names[model_out]} - tested: {cnt} / {len(datasets[0])}")

        if simple_ensemble:
            pred_no_back = int(mapping[pred_no_back])
            if label == pred_no_back:
                correct_no_back += 1
            print(f"\tSimple Ens accuracy: {correct_no_back}/{cnt} - predicted class {pred_no_back}: {class_names[pred_no_back]} - tested: {cnt} / {len(datasets[0])}")

        if label == prediction:
            correct += 1
            
        print(f"\tEnsemble accuracy: {correct}/{cnt} - predicted class {prediction}: {class_names[prediction]} - tested: {cnt} / {len(datasets[0])}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #--------------------ImageNet-A--------------------
    imageNetA = True
    naug = 6
    top = 1
    niter = 1
    testSingleModels = True
    simple_ensemble = True

    #set the seed
    torch.manual_seed(0)
    np.random.seed(0)

    #ENS
    models_type = ["memo", "tpt"]
    args = [
        {"device": "cuda", "naug": naug, "A": imageNetA, "drop": 0, "ttt_steps": 1, "model": "RN50", "top": top},
        {"device": "cuda", "naug": naug, "A": imageNetA, "ttt_steps": 1, "align_steps": 0, "arch": "RN50", "top": top}
        ]
    temps = [1.55, 0.7]
    names = ["MEMO", "TPT"]

    models = []
    datasets = []
    mapping = None
    load_model = {
        "memo": memo,
        "tpt": TPT
    }
    for i in range(len(models_type)):
        model, data, mapping = load_model[models_type[i]](**args[i])
        models.append(model)
        datasets.append(data)

    print("Testing on ImageNet-A")
    torch.autograd.set_detect_anomaly(True)
    test(models=models, datasets=datasets, temps=temps, mapping=mapping, names=names, 
         device=device, niter=niter, top=top, simple_ensemble=simple_ensemble, testSingleModels=testSingleModels)
    
    for i in range(len(models)):
        del models[i]
        del datasets[i]
    
    #--------------------ImageNet-V2--------------------
    imageNetA = False
    naug = 32
    top = 0.1
    niter = 1
    testSingleModels = False
    simple_ensemble = True

    #set the seed
    torch.manual_seed(0)
    np.random.seed(0)

    #ENS
    models_type = ["memo", "tpt"]
    args = [
        {"device": "cuda", "naug": naug, "A": imageNetA, "drop": 0, "ttt_steps": 1, "model": "RN50", "top": top},
        {"device": "cuda", "naug": naug, "A": imageNetA, "ttt_steps": 1, "align_steps": 0, "arch": "RN50", "top": top},
        ]
    temps = [1.55, 0.7]
    names = ["MEMO", "TPT"]

    models = []
    datasets = []
    mapping = None
    load_model = {
        "memo": memo,
        "tpt": TPT
    }
    for i in range(len(models_type)):
        model, data, mapping = load_model[models_type[i]](**args[i])
        models.append(model)
        datasets.append(data)

    print("Testing on ImageNet-V2")
    torch.autograd.set_detect_anomaly(True)
    test(models=models, datasets=datasets, temps=temps, mapping=mapping, names=names, 
         device=device, niter=niter, top=top, simple_ensemble=simple_ensemble, testSingleModels=testSingleModels)

if __name__ == "__main__":
    main()