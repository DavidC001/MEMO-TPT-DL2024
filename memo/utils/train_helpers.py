import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import sys

sys.path.append('.')

# https://github.com/bethgelab/robustness/blob/main/robusta/batchnorm/bn.py#L175
def _modified_bn_forward(self, input):
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return nn.functional.batch_norm(input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps)


def build_model(model_name, device, prior_strength=-1):
    if model_name == 'resnext':
        net = models.resnext101_32x8d().to(device=device)
    else:
        net = models.resnet50()
    net = torch.nn.DataParallel(net).to(device=device)

    if prior_strength >= 0:
        print('modifying BN forward pass')
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)
        nn.BatchNorm2d.forward = _modified_bn_forward
    return net
