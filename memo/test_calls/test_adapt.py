import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import copy

sys.path.append('.')
cudnn.benchmark = True
from memo.utils.adapt_helpers import memo_adapt_single, memo_test_single
from memo.utils.train_helpers import memo_build_model
from dataloaders.dataloader import get_dataloaders
from memo.utils.adapt_helpers import memo_transforms

# TODO: change to passed device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def memo_marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def memo_test_adapt(model_name, batch_size, lr, weight_decay, opt, niter, prior_strength):
    net = memo_build_model(model_name=model_name, device=DEVICE, prior_strength=prior_strength)

    imageNet_A, imageNet_V2 = get_dataloaders('datasets', memo_transforms)

    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    if opt == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    print('Running...')
    correct = []
    for i in tqdm(range(len(imageNet_A))):
        net2 = copy.deepcopy(net)
        data = imageNet_A[i]
        image = data["img"]
        label = int(data["label"])
        memo_adapt_single(net2, image, optimizer, memo_marginal_entropy, niter, batch_size, prior_strength, DEVICE)
        correct.append(memo_test_single(net2, image, label, prior_strength, DEVICE)[0])
        if (i % 100 == 0): print(f'\nMEMO adapt test error A {(1 - np.mean(correct)) * 100:.2f}')

    print(f'MEMO adapt test error A {(1 - np.mean(correct)) * 100:.2f}')

    correct = []
    for i in tqdm(range(len(imageNet_V2))):
        net2 = copy.deepcopy(net)
        data = imageNet_V2[i]
        image = data["img"]
        label = int(data["label"])
        memo_adapt_single(net2, image, optimizer, memo_marginal_entropy, niter, batch_size, prior_strength, DEVICE)
        correct.append(memo_test_single(net2, image, label, prior_strength, DEVICE)[0])

    print(f'MEMO adapt test error V2 {(1 - np.mean(correct)) * 100:.2f}')

