from EasyTPT.utils import tpt_get_datasets
from EasyTPT.models import EasyTPT

import torch
import torch.nn as nn

def TPT(device="cuda", naug=64, arch="RN50", A=True, ttt_steps=1, align_steps=0, top=0.1):
    """
    Return the TPT model initialized with the given parameters

    Args:
        - device: device to use - default: cuda
        - naug: number of augmentations to use - default: 64
        - arch: backbone model to use - default: RN50
        - A: use ImageNet A or ImageNet V2 - default: True
        - ttt_steps: number of iterations for the TTT - default: 1
        - align_steps: number of iterations for the alignment of the image embeddings - default: 0
        - top: top confidence to select the augmented samples - default: 0.1
    """
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

def memo(device="cuda", prior_strength=0.94, naug=64, A=True, drop=0, ttt_steps=1, model="RN50", top=0.1):
    """
    Return the MEMO model initialized with the given parameters

    Args:
        - device: device to use - default: cuda
        - prior_strength: strength of the prior for the BN layers - default: 0.94
        - naug: number of augmentations to use - default: 64
        - A: use ImageNet A or ImageNet V2 - default: True
        - drop: dropout to use, by setting it to >0 the model will use the ensemble strategy - default: 0
        - ttt_steps: number of iterations for the TTT - default: 1
        - model: backbone model to use - default: RN50
        - top: top confidence to select the augmented samples - default: 0.1
    """
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
