import torch
from torch import nn

class Ensemble(nn.Module):
    def __init__(self, models, temps, device="cuda", test_single_models=False):
        super(Ensemble, self).__init__()
        self.models = models
        self.temps = temps
        self.test_single_models = test_single_models
        self.device = device

    def entropy(self, logits):
        return -(torch.exp(logits) * logits).sum(dim=-1)

    def marginalDistribution(self, models_logits):
        # average logits for each model
        avg_models_logits = torch.Tensor(models_logits.shape[0], models_logits.shape[2]).to(self.device)
        for i, model_logits in enumerate(models_logits):
            avg_outs = torch.logsumexp(model_logits, dim=0) - torch.log(torch.tensor(model_logits.shape[0]))
            min_real = torch.finfo(avg_outs.dtype).min
            avg_outs = torch.clamp(avg_outs, min=min_real)
            avg_outs /= self.temps[i]
            avg_models_logits[i] = torch.log_softmax(avg_outs, dim=0)

        with torch.no_grad():
            entropies = torch.stack([self.entropy(logits) for logits in avg_models_logits])
            sum_entropies = torch.sum(entropies, dim=0)
            scale = [entopy / sum_entropies for entopy in entropies]
        
        avg_logits = torch.sum(torch.stack([scale[i].item() * avg_models_logits[i] for i in range(len(avg_models_logits))]), dim=0)

        return avg_logits

    def forward(self, inputs, niter=1, top=0.1):
        model_outs = None
        if self.test_single_models:
            model_outs = [model.predict(inputs[i], niter=niter) for i, model in enumerate(self.models)]

        for i in range(niter):
            outs = torch.stack([model(inputs[i], top).to(self.device) for i, model in enumerate(self.models)]).to(self.device)
            avg_logit = self.marginalDistribution(outs)

            loss = self.entropy(avg_logit)
            loss.backward()
            for model in self.models:
                model.optimizer.zero_grad()
                model.optimizer.step()
        
        outs = torch.stack([model(inputs[i], top).to(self.device) for i, model in enumerate(self.models)]).to(self.device)
        avg_logit = self.marginalDistribution(outs)
        prediction = torch.argmax(avg_logit, dim=0)

        return model_outs, prediction
    
    def reset(self):
        for model in self.models:
            model.reset()

        




