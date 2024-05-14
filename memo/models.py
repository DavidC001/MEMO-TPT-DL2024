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
from copy import deepcopy

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
    def __init__(self, net, device, classes_mask, prior_strength=1, lr=0.005, weight_decay=0.0001, opt='sgd', niter=1,
                 top=0.1):
        super(EasyMemo, self).__init__()

        self.device = device
        self.prior_strength = prior_strength
        self.net = net
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

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x).to(self.device)
            print(f"Shape forward: {x.shape}")
            logits = self.inference(x)
            logits, self.confidence_idx = self.topk_selection(logits)
        else:
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            x = x.to(self.device)
            logits = self.inference(x)
        return logits

    def inference(self, x):
        self.net.eval()
        outputs = self.net(x)

        out_app = torch.zeros(outputs.shape[0], len(self.classes_mask))
        for i, out in enumerate(outputs):
            out_app[i] = out[self.classes_mask]
        return out_app

    def predict(self, x):
        """
        Predicts the class of the input x
        Args:
            x: Tensor of shape (N, C, H, W)

        Returns: The predicted classes

        """
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
            predicted = outputs.argmax(1).item()

        return predicted

    def reset(self):
        del self.optimizer
        self.optimizer = self.memo_optimizer_model(lr=self.lr, weight_decay=self.weight_decay, opt=self.opt)
        self.confidence_idx = None
        self.net.load_state_dict(deepcopy(self.initial_state))

    def memo_modify_bn_pass(self):
        print('modifying BN forward pass')
        nn.BatchNorm2d.prior = self.prior_strength
        nn.BatchNorm2d.forward = _modified_bn_forward

    def memo_optimizer_model(self, lr=0.005, weight_decay=0.0001, opt='sgd'):
        optimizer = optim.SGD(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        if opt == 'adamw':
            optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
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


def memo_get_datasets(augmix: True, augs=64):
    """
    Returns the ImageNetA and ImageNetV2 datasets for the memo model
    Args:
        augmix: Whether to use AugMix or not
        augs: The number of augmentations to compute. Must be greater than 1

    Returns:

    """
    assert augs > 1, 'The number of augmentations must be greater than 1'
    memo_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = EasyAgumenter(memo_transforms, preprocess, augmix, augs - 1)
    imageNet_A, imageNet_V2 = get_dataloaders('datasets', transform)
    return imageNet_A, imageNet_V2


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imageNet_A, imageNet_V2 = memo_get_datasets(augmix=False, augs=2)

    mapping_a = [int(x) for x in imageNet_A.classnames.keys()]
    memo = EasyMemo(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device), device, mapping_a,
                    prior_strength=0.94, top=0.5)
    correct = []
    for i in range(len(imageNet_A)):
        data = imageNet_A[i]
        image = data["img"]
        label = int(data["label"])
        logit = memo.forward(image)
        predict = memo.predict(image)
        print(logit.shape, predict)
        memo.reset()
        exit(0)
    print("Accuracy: ", sum(correct) / len(correct))
