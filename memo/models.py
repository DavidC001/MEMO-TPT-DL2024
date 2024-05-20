import time

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import sys
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy

sys.path.append('.')
from memo.utils import memo_get_datasets
from EasyModel import EasyModel


def _modified_bn_forward(self, input):
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return nn.functional.batch_norm(input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps)


class EasyMemo(EasyModel):
    """
    A class to wrap a neural network with the MEMO TTA method
    """

    def __init__(self, net, device, classes_mask, prior_strength: float = 1.0, lr=0.005, weight_decay=0.0001, opt='sgd',
                 niter=1, top=0.1, drop=False):
        """
        Initializes the EasyMemo model with various arguments
        Args:
            net: The model to wrap with EasyMemo
            device: The device to run the model on(usually 'CPU' or 'CUDA')
            classes_mask: The classes to consider for the model(used for Imagenet-A)
            prior_strength: The strength of the prior to use in the modified BN forward pass
            lr: The Learning rate for the optimizer of the model
            weight_decay: The weight decay for the optimizer of the model
            opt: Which optimizer to use for this model between 'sgd' and 'adamw' for the respective optimizers
            niter: The number of iterations to run the memo pass for
            top: The percentage of the top logits to consider for confidence selection
        """
        super(EasyMemo, self).__init__()

        self.drop = drop
        self.device = device
        self.prior_strength = prior_strength
        self.net = net.to(device)
        self.optimizer = self.get_optimizer(lr=lr, weight_decay=weight_decay, opt=opt)
        self.lr = lr
        self.weight_decay = weight_decay
        self.opt = opt
        self.confidence_idx = None
        self.memo_modify_bn_pass()
        self.criterion = self.avg_entropy
        self.niter = niter
        self.top = top
        self.initial_state = deepcopy(self.net.state_dict())
        self.classes_mask = classes_mask

    def forward(self, x, top=-1):
        """
        Forward pass where we check which type of input we have and we call the inference on the input image Tensor
        Args:
            top: How many samples to select from the batch
            x: A Tensor of shape (N, C, H, W) or a list of Tensors of shape (N, C, H, W)

        Returns: The logits after the inference pass

        """
        self.top = top if top > 0 else self.top
        # print(f"Shape forward: {x.shape}")
        if isinstance(x, list):
            x = torch.stack(x).to(self.device)
            # print(f"Shape forward: {x.shape}")
            logits = self.inference(x)
            logits, self.confidence_idx = self.select_confident_samples(logits, self.top)
        else:
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            x = x.to(self.device)
            logits = self.inference(x)

        # print(f"[EasyMemo] input shape: {x.shape}")
        # print(f"[EasyMemo] logits shape: {logits.shape}")
        return logits

    def inference(self, x):
        """
        Return the logits of the image in input x
        Args:
            x: A Tensor of shape (N, C, H, W) of an Image

        Returns: The logits for that Tensor image

        """
        if self.drop:
            self.net.train()
        else:
            self.net.eval()
        outputs = self.net(x)

        out_app = torch.zeros(outputs.shape[0], len(self.classes_mask)).to(self.device)
        for i, out in enumerate(outputs):
            out_app[i] = out[self.classes_mask]
        return out_app

    def predict(self, x, niter=1):
        """
        Predicts the class of the input x, which is an image
        Args:
            niter: The number of iteration on which to run the memo pass
            x: Tensor of shape (N, C, H, W)

        Returns: The predicted classes

        """
        self.niter = niter
        if self.drop:
            self.net.train()
            predicted = self.predict_dropout(x)
        else:
            self.net.eval()
            for iteration in range(self.niter):
                self.optimizer.zero_grad()
                outputs = self.forward(x)
                loss = self.criterion(outputs)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                outputs = self.net(x[0].unsqueeze(0).to(self.device))
                outs = torch.zeros(outputs.shape[0], len(self.classes_mask)).to(self.device)
                for i, out in enumerate(outputs):
                    outs[i] = out[self.classes_mask]
                predicted = outs.argmax(1).item()

        return predicted

    def reset(self):
        """Resets the model to its initial state"""
        del self.optimizer
        self.optimizer = self.get_optimizer(lr=self.lr, weight_decay=self.weight_decay, opt=self.opt)
        self.confidence_idx = None
        self.net.load_state_dict(deepcopy(self.initial_state))

    def memo_modify_bn_pass(self):
        print('modifying BN forward pass')
        nn.BatchNorm2d.prior = self.prior_strength
        nn.BatchNorm2d.forward = _modified_bn_forward

    def get_optimizer(self, lr=0.005, weight_decay=0.0001, opt='sgd'):
        """
        Initializes the optimizer for the memo model
        Args:
            lr: The learning rate for the optimizer
            weight_decay: The weight decay for the optimizer
            opt: Which optimizer to use

        Returns: The optimizer for the memo model

        """
        if opt == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt == 'adamw':
            optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Invalid optimizer selected')
        return optimizer

    def predict_dropout(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            prediction = outputs.sum(0).argmax().item()

        return prediction


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imageNet_A, imageNet_V2 = memo_get_datasets('augmix', 64)
    mapping_a = [int(x) for x in imageNet_A.classnames.keys()]
    mapping_V2 = [int(x) for x in imageNet_V2.classnames.keys()]

    dataset = imageNet_A
    mapping = mapping_a
    net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    net.layer4.add_module('dropout', nn.Dropout(0.5, inplace=True))

    memo = EasyMemo(net.to(device), device, mapping, prior_strength=1, top=0.1)

    np.random.seed(0)
    torch.manual_seed(0)

    correct = 0
    cnt = 0
    index = np.random.permutation(range(len(dataset)))
    iterate = tqdm(index)
    for i in iterate:
        data = dataset[i]
        image = data["img"]
        label = int(data["label"])
        dropout_prediction = memo.predict(image)
        memo.reset()
        correct+=mapping[dropout_prediction] == label
        cnt+=1
        iterate.set_description(desc=f"Current accuracy {correct / cnt:.2f}")
    # print("Accuracy: ", sum(correct) / len(correct))
