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
from memo.utils.adapt_helpers import adapt_single, test_single
from memo.utils.train_helpers import build_model
from dataloaders.dataloader import get_dataloaders
from memo.utils.adapt_helpers import te_transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def test_adapt(model_name, batch_size, lr, weight_decay, opt, niter, prior_strength):
    net = build_model(model_name, DEVICE)

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
        adapt_single(net2, image, optimizer, marginal_entropy, niter, batch_size, prior_strength, DEVICE)
        correct.append(test_single(net2, image, label, prior_strength, DEVICE)[0])

    print(f'MEMO adapt test error {(1 - np.mean(correct)) * 100:.2f}')
