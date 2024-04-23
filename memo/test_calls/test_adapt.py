import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import copy

cudnn.benchmark = True
from utils.adapt_helpers import adapt_single, test_single
from utils.train_helpers import build_model
from dataloaders.dataloader import get_dataloaders
from utils.adapt_helpers import te_transforms


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def test_adapt():
    model_name = 'resnet'
    batch_size = 64
    lr = 0.00025 if model_name == 'resnet' else 0.0001
    weight_decay = 0 if model_name == 'resnet' else 0.01
    opt = 'SGD' if model_name == 'resnet' else 'adamw'
    niter = 1
    prior_strength = -1

    net = build_model(model_name)

    imageNet_A, imageNet_V2 = get_dataloaders('datasets', te_transforms)

    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    if opt == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    print('Running...')
    correct = []
    for i in tqdm(range(len(imageNet_A))):
        net2 = copy.deepcopy(net)
        data = imageNet_A[i]
        image = data["img"]
        label = data["label"]
        adapt_single(net2, image, optimizer, marginal_entropy, niter, batch_size, prior_strength)
        correct.append(test_single(net2, image, label, prior_strength)[0])

    print(f'MEMO adapt test error {(1 - np.mean(correct)) * 100:.2f}')
