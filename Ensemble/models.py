import torch
from torch import nn
from math import floor
import sys
sys.path.append(".")
from EasyModel import EasyModel

class Ensemble(nn.Module):
    """
    Ensemble class. Implements an ensemble of models with entropy minimization.

    Attributes:
        models (list[EasyModel]): A list of models to be used in the ensemble.
        temps (list): A list of temperature values corresponding to each model.
        test_single_models (bool): Whether to test each individual model in addition to the ensemble.
        simple_ensemble (bool): Whether to perform the entropy minimization step.
        device (str): The device to be used for computation.
    """
    def __init__(self, models:list[EasyModel], temps, device="cuda", test_single_models=False, simple_ensemble=False):
        """
        Initializes an Ensemble object.

        Args:
            models (list[EasyModel]): A list of models to be used in the ensemble.
            temps (list): A list of temperature values corresponding to each model.
            device (str, optional): The device to be used for computation. Defaults to "cuda".
            test_single_models (bool, optional): Whether to test each individual model in addition to the ensemble. Defaults to False.
            simple_ensemble (bool, optional): Whether to perform the entropy minimization step. Defaults to False.
        """
        super(Ensemble, self).__init__()
        self.models = models
        self.temps = temps
        self.test_single_models = test_single_models
        self.device = device
        self.simple_ensemble = simple_ensemble

    def entropy(self, logits):
        """
        Computes the entropy of a set of logits.

        Args:
            logits (torch.Tensor): The logits to compute the entropy of.

        Returns:
            torch.Tensor: The entropy of the logits.
        """
        return -(torch.exp(logits) * logits).sum(dim=-1)

    def marginal_distribution(self, models_logits):
        """
        Computes the marginal distribution of the ensemble.

        Args:
            models_logits (torch.Tensor): The logits of the models in the ensemble.

        Returns:
            torch.Tensor: The marginal distribution of the ensemble.
        """
        # average logits for each model
        avg_models_distributions = torch.Tensor(models_logits.shape[0], models_logits.shape[2]).to(self.device)
        for i, model_logits in enumerate(models_logits):
            avg_outs = torch.logsumexp(model_logits, dim=0) - torch.log(torch.tensor(model_logits.shape[0]))
            min_real = torch.finfo(avg_outs.dtype).min
            avg_outs = torch.clamp(avg_outs, min=min_real)
            avg_outs /= self.temps[i]
            avg_models_distributions[i] = torch.log_softmax(avg_outs, dim=0)

        with torch.no_grad():
            entropies = torch.stack([self.entropy(logits) for logits in avg_models_distributions]).to(self.device)
            sum_entropies = torch.sum(entropies, dim=0)
            scale = torch.stack([sum_entropies/entopy for entopy in entropies]).to(self.device)
            #normalize sum to 1
            scale = scale / torch.sum(scale)

        # print("\t\t[Ensemble] Entropies: ", entropies)
        # print("\t\t[Ensemble] Scales: ", scale)

        marginal_dist = torch.sum(torch.stack([scale[i].item() * avg_models_distributions[i] for i in range(len(avg_models_distributions))]), dim=0)

        return marginal_dist

    def get_models_outs(self, inputs, top=0.1):
        """
        Computes the outputs of the models in the ensemble.

        Args:
            inputs (list): A list of inputs to be fed to the models.
            top (float, optional): The top percentage of the outputs to be used. Defaults to 0.1.

        Returns:
            torch.Tensor: The outputs of the models in the ensemble.
        """
        model_outs = torch.stack([model(inputs[i], top).to(self.device) for i, model in enumerate(self.models)]).to(self.device)
        return model_outs.to(self.device)

    def get_models_predictions(self, inputs):
        """
        Computes the predictions of the single models in the ensemble.

        Args:
            inputs (list): A list of inputs to be fed to the models.

        Returns:
            list: A list of the predictions of the single models.
        """
        models_pred = [model.predict(inputs[i]) for i, model in enumerate(self.models)]
        return models_pred

    def entropy_minimization(self, inputs, niter=1, top=0.1):
        """
        Test time adaptation step. Minimizes the entropy of the ensemble's predictions.

        Args:
            inputs (list): A list of inputs to be fed to the models.
            niter (int, optional): The number of iterations to perform. Defaults to 1.
            top (float, optional): The top percentage of the outputs to be used. Defaults to 0.1.
        """
        for i in range(niter):
            outs = self.get_models_outs(inputs, top)
            avg_logit = self.marginal_distribution(outs)

            loss = self.entropy(avg_logit)
            loss.backward()
            for model in self.models:
                model.optimizer.step()
                model.optimizer.zero_grad()

    def forward(self, inputs, niter=1, top=0.1):
        """
        Forward pass of the ensemble.

        Args:
            inputs (list): A list of inputs to be fed to the models.
            niter (int, optional): The number of iterations to perform. Defaults to 1.
            top (float, optional): The top percentage of the outputs to be used. Defaults to 0.1.

        Returns:
            list, torch.Tensor, torch.Tensor: The predictions of the single models, the prediction of the ensemble without the entropy minimization step, and the prediction of the ensemble.
        """
        # get models outputs
        self.reset()
        models_pred = self.get_models_predictions(inputs)

        self.reset()
        self.entropy_minimization(inputs, niter, top)
            
        with torch.no_grad():
            outs = self.get_models_outs([i[0] for i in inputs], top)
            avg_logit = self.marginal_distribution(outs)
            prediction = torch.argmax(avg_logit, dim=0)

        if self.simple_ensemble:
            self.reset()    
            for model in self.models: model.eval()
            outs = self.get_models_outs(inputs, top)
            avg_logit = self.marginal_distribution(outs)
            prediction_no_back = torch.argmax(avg_logit, dim=0)

        return models_pred, prediction_no_back, prediction

    def reset(self):
        """
        Resets the models in the ensemble.
        """
        for model in self.models:
            model.reset()

        




