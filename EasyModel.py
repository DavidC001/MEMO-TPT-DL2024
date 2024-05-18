from torch import nn
import torch
import numpy as np

class EasyModel(nn.Module):
    def __init__(self):
        super(EasyModel, self).__init__()

    def select_confident_samples(self, logits, top):
        """
        Performs confidence selection, will return the indexes of the
        augmentations with the highest confidence as well as the filtered
        logits

        Parameters:
        - logits (torch.Tensor): the logits of the model [NAUGS, NCLASSES]
        - top (float): the percentage of top augmentations to use

        Returns:
        - logits (torch.Tensor): the filtered logits of the model [NAUGS*top, NCLASSES]
        """
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        idx = torch.argsort(batch_entropy, descending=False)[
            : int(batch_entropy.size()[0] * top)
        ]
        return logits[idx], idx
    
    def avg_entropy(self, outputs):
        """
        Computes the average entropy of the model outputs

        Parameters:
        - outputs (torch.Tensor): the logits of the model [NAUGS, NCLASSES]
        
        Returns:
        - avg_entropy (torch.Tensor): the average entropy of the model outputs [1]
        """
        logits = outputs - outputs.logsumexp(
            dim=-1, keepdim=True
        )  # logits = outputs.log_softmax(dim=1) [N, 1000]
        avg_logits = logits.logsumexp(dim=0) - np.log(
            logits.shape[0]
        )  # avg_logits = logits.mean(0) [1, 1000]
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

    def forward(self, x):
        return super(EasyModel, self).forward(x)
        
    def predict(self, x):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
