import torch
from torch import nn
from math import floor

class Ensemble(nn.Module):
    def __init__(self, models, temps, device="cuda", test_single_models=False):
        super(Ensemble, self).__init__()
        self.models = models
        self.temps = temps
        self.test_single_models = test_single_models
        self.device = device

    def entropy(self, logits):
        return -(torch.exp(logits) * logits).sum(dim=-1)

    def marginal_distribution(self, models_logits):
        # average logits for each model
        avg_models_logits = torch.Tensor(models_logits.shape[0], models_logits.shape[2]).to(self.device)
        for i, model_logits in enumerate(models_logits):
            avg_outs = torch.logsumexp(model_logits, dim=0) - torch.log(torch.tensor(model_logits.shape[0]))
            min_real = torch.finfo(avg_outs.dtype).min
            avg_outs = torch.clamp(avg_outs, min=min_real)
            avg_outs /= self.temps[i]
            avg_models_logits[i] = torch.log_softmax(avg_outs, dim=0)

        with torch.no_grad():
            entropies = torch.stack([self.entropy(logits) for logits in avg_models_logits]).to(self.device)
            sum_entropies = torch.sum(entropies, dim=0)
            scale = torch.stack([sum_entropies/entopy for entopy in entropies]).to(self.device)
            #normalize sum to 1
            scale = scale / torch.sum(scale)

        print("\t\t[Ensemble] Entropies: ", entropies)
        print("\t\t[Ensemble] Scales: ", scale)

        avg_logits = torch.sum(torch.stack([scale[i].item() * avg_models_logits[i] for i in range(len(avg_models_logits))]), dim=0)

        return avg_logits

    def get_models_outs(self, inputs, top=0.1):
        model_outs = torch.stack([model(inputs[i], top).to(self.device) for i, model in enumerate(self.models)]).to(self.device)
        return model_outs.to(self.device)

    def get_models_predictions(self, inputs):
        models_pred = [model.predict(inputs[i]) for i, model in enumerate(self.models)]
        return models_pred

    def entropy_minimization(self, inputs, niter=1, top=0.1):
        for i in range(niter):
            outs = self.get_models_outs(inputs, top)
            avg_logit = self.marginal_distribution(outs)

            loss = self.entropy(avg_logit)
            loss.backward()
            for model in self.models:
                model.optimizer.step()
                model.optimizer.zero_grad()

    def forward(self, inputs, niter=1, top=0.1):
        # get models outputs
        self.reset()
        models_pred = self.get_models_predictions(inputs)

        self.reset()
        self.entropy_minimization(inputs, niter, top)
        
        with torch.no_grad():
            outs = self.get_models_outs(inputs, top)
            avg_logit = self.marginal_distribution(outs)
            prediction = torch.argmax(avg_logit, dim=0)

        return models_pred, prediction

    def reset(self):
        for model in self.models:
            model.reset()

        




