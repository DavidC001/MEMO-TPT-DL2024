from dataloaders.dataloader import get_classes_names

from Ensemble.models import Ensemble
from Ensemble.utils import memo, TPT

import numpy as np
import torch


def test(models, datasets, temps, mapping, names,
         device="cuda", niter=1, top=0.1,
         simple_ensemble=False, testSingleModels=False, verbose=100):
    """
    Test the ensemble model on the datasets

    Args:
        - models: list of models
        - datasets: list of datasets
        - temps: list of temperatures for the models to rescale the logits
        - names: names of the models
        - mapping: mapping of the classes outputted by the models to the original 1000 classes
        - device: device to use
        - niter: number of iterations for the TTT
        - top: top confidence to select the augmented samples
        - simple_ensemble: use the simple ensemble, only marginalizing the ditributions
        - testSingleModels: test the single models inside the ensamble
        - verbose: verbosity level
    
    Returns:
        - results: dictionary with the results of the test
    """
    correct = 0
    correct_no_back = 0
    correctSingle = [0] * len(models)
    cnt = 0

    class_names = get_classes_names()

    # shuffle the data
    indx = np.random.permutation(range(len(datasets[0])))

    model = Ensemble(models, temps=temps,
                     device=device, test_single_models=testSingleModels,
                     simple_ensemble=simple_ensemble)
    print("Ensemble model created starting TTA, samples:", len(indx))
    for i in indx:
        cnt += 1
        data = [datasets[j][i]["img"] for j in range(len(datasets))]

        labels = [datasets[j][i]["label"] for j in range(len(datasets))]
        # check if the labels are the same
        assert all(x == labels[0] for x in labels), "Labels are not the same"
        label = labels[0]
        name = datasets[0][i]["name"]

        if(cnt%verbose==0): print(f"Tested Samples: {cnt} / {len(datasets[0])} - current sample: {name}")

        models_out, pred_no_back, prediction = model(data, niter=niter, top=top)
        models_out = [int(mapping[model_out]) for model_out in models_out]
        prediction = int(mapping[prediction])

        if testSingleModels:
            for i, model_out in enumerate(models_out):
                if label == model_out:
                    correctSingle[i] += 1

                if(cnt%verbose==0): 
                    print(
                    f"\t{names[i]} model accuracy: {correctSingle[i]}/{cnt} - predicted class {model_out}: {class_names[model_out]} - tested: {cnt} / {len(datasets[0])}")

        if simple_ensemble:
            pred_no_back = int(mapping[pred_no_back])
            if label == pred_no_back:
                correct_no_back += 1
            if(cnt%verbose==0): 
                print(
                f"\tSimple Ens accuracy: {correct_no_back}/{cnt} - predicted class {pred_no_back}: {class_names[pred_no_back]} - tested: {cnt} / {len(datasets[0])}")

        if label == prediction:
            correct += 1

        if(cnt%verbose==0): 
            print(
            f"\tEnsemble accuracy: {correct}/{cnt} - predicted class {prediction}: {class_names[prediction]} - tested: {cnt} / {len(datasets[0])}")
    
    results = {
        "Ensemble": correct / len(datasets[0]),
        "Simple Ens": correct_no_back / len(datasets[0]),
        "Single Models": [correctSingle[i] / len(datasets[0]) for i in range(len(models))]
    }


# expand args
def runTest(models_type, args, temps, names, naug=64, niter=1, top=0.1, device="cuda", simple_ensemble=False,
            testSingleModels=False, imageNetA=True, dataset_root="datasets", verbose=100):
    """
    Run the test on Ensemble model with the given arguments

    Args:
        - models_type: list of the models type to use
        - args: list of the arguments for the models
        - temps: list of temperatures for the models to rescale the logits
        - names: names of the models
        - naug: number of augmentations to use
        - niter: number of iterations for the TTT
        - top: top confidence to select the augmented samples
        - device: device to use
        - simple_ensemble: use the simple ensemble, only marginalizing the ditributions
        - testSingleModels: test the single models inside the ensamble
        - imageNetA: use ImageNet A or ImageNet V2
        - dataset_root: root of the dataset
        - verbose: verbosity level

    Returns:
        - results: dictionary with the results of the test
    """

    models = []
    datasets = []
    mapping = None
    load_model = {
        "memo": memo,
        "tpt": TPT
    }
    for i in range(len(models_type)):
        model, data, mapping = load_model[models_type[i]](**args[i], A=imageNetA, top=top, naug=naug,
                                                          dataset_root=dataset_root)
        models.append(model)
        datasets.append(data)

    result = test(models=models, datasets=datasets, temps=temps, mapping=mapping, names=names,
         device=device, niter=niter, top=top, simple_ensemble=simple_ensemble, testSingleModels=testSingleModels,
         verbose=verbose)

    for model in models:
        del model
    
    return result
