import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import sys
import numpy as np
import torch.optim as optim
from copy import deepcopy

sys.path.append('.')
from memo.utils import memo_get_datasets


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
        if self.drop:
            net.layer4.add_module('dropout', nn.Dropout(0.5, inplace=True))
        self.device = device
        self.prior_strength = prior_strength
        self.net = net.to(device)
        self.optimizer = self.memo_optimizer_model(lr=lr, weight_decay=weight_decay, opt=opt)
        self.lr = lr
        self.weight_decay = weight_decay
        self.opt = opt
        self.confidence_idx = None
        self.memo_modify_bn_pass()
        self.criterion = memo_marginal_entropy
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
            logits, self.confidence_idx = self.topk_selection(logits)
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
        else:
            self.net.eval()

        for iteration in range(self.niter):
            self.optimizer.zero_grad()
            outputs = self.forward(x)
            outputs, _ = self.topk_selection(outputs)
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
        self.optimizer = self.memo_optimizer_model(lr=self.lr, weight_decay=self.weight_decay, opt=self.opt)
        self.confidence_idx = None
        self.net.load_state_dict(deepcopy(self.initial_state))

    def memo_modify_bn_pass(self):
        print('modifying BN forward pass')
        nn.BatchNorm2d.prior = self.prior_strength
        nn.BatchNorm2d.forward = _modified_bn_forward

    def memo_optimizer_model(self, lr=0.005, weight_decay=0.0001, opt='sgd'):
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

    def memo_adapt_single(self, inputs):
        """
        A single step of memo adaptation
        Args:
            inputs: A tensor of shape (N, C, H, W)

        """
        self.net.eval()
        assert self.niter > 0 and isinstance(self.niter, int), 'niter must be a positive integer'
        for iteration in range(self.niter):
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs)
            loss.backward()
            self.optimizer.step()

    def memo_test_single(self, image, label):
        """
        Tests the model on a single image and returns the correctness and confidence
        Args:
            image: A tensor of shape (N, C, H, W)
            label: The correct label for the test

        Returns: The correctness and confidence of the prediction

        """
        with torch.no_grad():
            outputs = self.net(image.to(device=self.device))
            _, predicted = outputs.max(1)
            confidence = nn.functional.softmax(outputs, dim=1).squeeze()[predicted].item()
        correctness = 1 if predicted.item() == label else 0
        return correctness, confidence

    def topk_selection(self, logits):
        """
        Selects the top k logits based on the batch entropy
        Args:
            logits: A tensor of shape (N, C)

        Returns: The filtered logits and the indices of the selected logits

        """
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        selected_idx = torch.argsort(batch_entropy, descending=False)[: int(batch_entropy.size()[0] * self.top)]
        return logits[selected_idx], selected_idx

    def dropout_train(self, x):
        self.net.train()
        outputs = self.forward(x)
        outputs, _ = self.topk_selection(outputs)

        return outputs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imageNet_A, imageNet_V2 = memo_get_datasets('identity',64, True)

    mapping_a = [int(x) for x in imageNet_A.classnames.keys()]
    net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # print(net2)
    # exit(0)

    memo = EasyMemo(net.to(device), device, mapping_a, prior_strength=0.94, top=1)
    correct = []
    for i in range(len(imageNet_A)):
        data = imageNet_A[i]
        # print(data['img'].shape);
        image = data["img"]
        label = int(data["label"])
        # logit = memo.forward(image)
        # predict = memo.predict(image)
        # print(logit.shape, predict)
        # predict = memo.forward(image)
        original = memo.forward(image)
        dropout = memo.dropout_train(image)
        memo.reset()
        print(original.shape, dropout.shape)
    # print("Accuracy: ", sum(correct) / len(correct))