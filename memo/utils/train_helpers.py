import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from dataloaders.dataloader import get_dataloaders

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
te_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize])


# https://github.com/bethgelab/robustness/blob/main/robusta/batchnorm/bn.py#L175
def _modified_bn_forward(self, input):
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return nn.functional.batch_norm(input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps)


def build_model(model_name, prior_strength=-1):
    if model_name == 'resnext':
        net = models.resnext101_32x8d().cuda()
    else:
        net = models.resnet50().cuda()
    net = torch.nn.DataParallel(net)

    if prior_strength >= 0:
        print('modifying BN forward pass')
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)
        nn.BatchNorm2d.forward = _modified_bn_forward
    return net


def prepare_loader(batch_size, use_transforms=True):
    te_transforms_local = te_transforms if use_transforms else None
    collate_fn = None if use_transforms else lambda x: x
    imageNet_A, imageNet_V2 = get_dataloaders('dataset', te_transforms_local)
    teloader = torch.utils.data.DataLoader(imageNet_A, batch_size, shuffle=False,
                                           num_workers=8, pin_memory=True, collate_fn=collate_fn)
    return imageNet_A, teloader
