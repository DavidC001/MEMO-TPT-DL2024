import torch
import torch.nn as nn
from torchvision.transforms import v2
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.models as models
import sys
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import copy

sys.path.append('.')
from dataloaders.dataloader import get_classes_names, get_dataloaders
from EasyTPT.utils import EasyAgumenter


def memo_marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def _modified_bn_forward(self, input):
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return nn.functional.batch_norm(input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps)


class EasyMemo(nn.Module):
    def __init__(self, net, device, prior_strength=1, lr=0.005, weight_decay=0.0001, opt='sgd', niter=1):
        super(EasyMemo, self).__init__()

        self.device = device
        self.prior_strength = prior_strength
        self.net = net
        self.optimizer = self.memo_optimizer_model(lr=lr, weight_decay=weight_decay, opt=opt)
        self.names = get_classes_names()
        self.memo_modify_bn_pass()
        self.criterion = memo_marginal_entropy
        self.niter = niter

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x).to(self.device)
            logits = self.inference(x)
            # Use confidence selection?
            # if self.selected_idx is not None:
            #     logits = logits[self.selected_idx]
            # else:
            #     logits, self.selected_idx = self.select_confident_samples(logits, 0.10)
        else:
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            x = x.to(self.device)
            logits = self.inference(x)
        return logits

    def inference(self, x):
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(x)
        return outputs

    def memo_modify_bn_pass(self):
        print('modifying BN forward pass')
        nn.BatchNorm2d.prior = self.prior_strength
        nn.BatchNorm2d.forward = _modified_bn_forward

    def memo_optimizer_model(self, lr=0.005, weight_decay=0.0001, opt='sgd'):
        optimizer = optim.SGD(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        if opt == 'adamw':
            optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

    def memo_adapt_single(self, inputs, niter):
        self.net.eval()
        assert niter > 0 and isinstance(niter, int), 'niter must be a positive integer'
        for iteration in range(niter):
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs)
            loss.backward()
            self.optimizer.step()

    def memo_test_single(self, image, label):
        with torch.no_grad():
            outputs = self.net(image.to(device=self.device))
            _, predicted = outputs.max(1)
            # print( "Predicted: ", names[predicted.item()], " Label: ", names[label])
            confidence = nn.functional.softmax(outputs, dim=1).squeeze()[predicted].item()
        correctness = 1 if predicted.item() == label else 0
        return correctness, confidence


def memo_get_datasets(augmix: True, augs=64):
    memo_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = EasyAgumenter(memo_transforms, preprocess, augmix, augs - 1)
    imageNet_A, imageNet_V2 = get_dataloaders('datasets', transform)
    return imageNet_A, imageNet_V2
