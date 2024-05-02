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


def memo_build_model(model_name, device, prior_strength=1):
    if model_name == 'resnext':
        weights = models.ResNeXt101_32X8D_Weights.DEFAULT
        net = models.resnext101_32x8d(weights=weights).to(device=device)
    elif model_name == 'vit16b':
        weights = models.ViT_B_16_Weights.DEFAULT
        net = models.vit_b_16(weights=weights).to(device=device)
    elif model_name == 'vit14h':
        weights = models.ViT_H_14_Weights.DEFAULT
        net = models.vit_b_16(weights=weights).to(device=device)
    else:
        weights = models.ResNet50_Weights.DEFAULT
        net = models.resnet50(weights=weights).to(device=device)

    print('modifying BN forward pass')
    nn.BatchNorm2d.prior = prior_strength
    nn.BatchNorm2d.forward = _modified_bn_forward

    return net
