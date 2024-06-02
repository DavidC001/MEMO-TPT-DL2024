import sys
import torch
import time
import numpy as np
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import argparse
from datetime import timedelta

sys.path.append('.')
from memo.utils import memo_get_datasets
from memo.models import EasyMemo


def MEMO_testing_step(test, arguments):
    print(f"Starting {test} evaluation...")
    device = arguments["memo"]["device"]
    mapping = arguments["memo"]["mapping"]
    prior_strength = arguments["memo"]["prior_strength"]
    lr = arguments["memo"]["lr"]
    weight_decay = arguments["memo"]["weight_decay"]
    opt = arguments["memo"]["opt"]
    niter = arguments["memo"]["niter"]
    top = arguments["memo"]["top"]
    ensemble = arguments["memo"]["ensemble"]
    dataset_root = arguments["dataset"]["dataset_root"]
    naugs = arguments["dataset"]["naug"]
    aug_type = arguments["dataset"]["aug_type"]
    weights = True if "weights" in arguments.keys() else False

    np.random.seed(0)
    torch.manual_seed(0)

    weights = models.ResNet50_Weights.DEFAULT if weights else models.ResNet50_Weights.IMAGENET1K_V1
    net = models.resnet50(weights=weights).to(device)
    if "drop" in arguments.keys():
        net.layer4.add_module('dropout', nn.Dropout(arguments["drop"], inplace=True))

    model = EasyMemo(net, device, mapping, prior_strength=prior_strength, top=top, ensemble=ensemble, lr=lr,
                     weight_decay=weight_decay, opt=opt, niter=niter)
    imageNet_A, imageNet_V2 = memo_get_datasets(aug_type, naugs, dataset_root)
    dataset = imageNet_A if arguments['dataset']['imageNetA'] else imageNet_V2

    correct = 0
    cnt = 0
    index = np.random.permutation(range(len(dataset)))
    iterate = tqdm(index)
    for i in iterate:
        data = dataset[i]
        image = data["img"]
        label = int(data["label"])
        prediction = model.predict(image)
        model.reset()
        correct += mapping[prediction] == label
        cnt += 1
        iterate.set_description(desc=f"Current accuracy {(correct / cnt) * 100:.2f}")
    print("--------------------------------------------------------------")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead")
    else:
        print("CUDA is available, using GPU")

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run MEMO tests",
    )

    parser.add_argument(
        "-d",
        "--datasets-root",
        type=str,
        help="Root folder of all the datasets, default='datasets'",
        default='datasets',
        metavar="",
    )
    parser.add_argument(
        "--data-to-test",
        type=str,
        help="Which dataset to test between 'a', 'v2', 'both'",
        default="both",
        metavar="",
    )

    args = vars(parser.parse_args())
    DATASET_ROOT = args["datasets_root"]
    DATASET_TO_TEST = args["data_to_test"]

    imageNet_A, imageNet_V2 = memo_get_datasets('augmix', 1, args["datasets_root"])
    mapping_a = [int(x) for x in imageNet_A.classnames.keys()]
    mapping_v2 = [int(x) for x in imageNet_V2.classnames.keys()]

    del imageNet_A, imageNet_V2

    memo_tests = False
    drop_tests = True
    ensemble_tests = True
    baseline_tests = False

    base_test = {
        "memo": {
            "device": device,
            "prior_strength": 1.0,
            "lr": 0.005,
            "weight_decay": 0.0001,
            "opt": 'sgd',
            "niter": 1,
            "top": 1,
            "ensemble": False,
        },
        "dataset": {
            "imageNetA": True,
            "naug": 1,
            "dataset_root": DATASET_ROOT,
            "aug_type": "augmix",
        },
    }

    tests = {
        "Baseline ImageNetA": {
            "memo": {
                "mapping": mapping_a,
                "ensemble": True,
                "top": 1,
            },
            "dataset": {
                "imageNetA": True,
            },
            "run": baseline_tests and (DATASET_TO_TEST in ["a", "both"]),
        },
        "Baseline ImageNetV2": {
            "memo": {
                "mapping": mapping_v2,
                "ensemble": True,
                "top": 1,
            },
            "dataset": {
                "imageNetA": False,
            },
            "run": baseline_tests and (DATASET_TO_TEST in ["v2", "both"]),
        },
        "Baseline ImageNetA ResNet50 weights V1": {
            "memo": {
                "mapping": mapping_a,
                "ensemble": True,
                "top": 1,
            },
            "dataset": {
                "imageNetA": True,
            },
            "weights": "v1",
            "run": baseline_tests and (DATASET_TO_TEST in ["a", "both"]),
        },
        "Baseline ImageNetV2 ResNet50 weights V1": {
            "memo": {
                "mapping": mapping_v2,
                "ensemble": True,
                "top": 1,
            },
            "dataset": {
                "imageNetA": False,
            },
            "weights": "v1",
            "run": baseline_tests and (DATASET_TO_TEST in ["v2", "both"]),
        },
        "MEMO ImageNetA, without topk selection": {
            "memo": {
                "top": 1,
                "mapping": mapping_a,
                "prior_strength": 0.94
            },
            "dataset": {
                "imageNetA": True,
                "naug": 8,
                "aug_type": "cut",
            },
            "run": memo_tests and (DATASET_TO_TEST in ["a", "both"]),
        },
        "MEMO ImageNetV2, without topk selection": {
            "memo": {
                "top": 1,
                "mapping": mapping_v2,
                "prior_strength": 0.94
            },
            "dataset": {
                "imageNetA": False,
                "naug": 8,
                "aug_type": "cut",
            },
            "run": memo_tests and (DATASET_TO_TEST in ["v2", "both"]),
        },
        "MEMO ImageNetA, with topk selection": {
            "memo": {
                "top": 0.1,
                "mapping": mapping_a,
                "prior_strength": 0.94
            },
            "dataset": {
                "imageNetA": True,
                "naug": 64,
                "aug_type": "cut",
            },
            "run": memo_tests and (DATASET_TO_TEST in ["a", "both"]),
        },
        "MEMO ImageNetV2, with topk selection": {
            "memo": {
                "top": 0.1,
                "mapping": mapping_v2,
                "prior_strength": 0.94
            },
            "dataset": {
                "imageNetA": False,
                "naug": 64,
                "aug_type": "cut",
            },
            "run": memo_tests and (DATASET_TO_TEST in ["v2", "both"]),
        },
        "DROP ImageNetA, without topk selection": {
            "memo": {
                "top": 1,
                "mapping": mapping_a,
                "ensemble": True,
            },
            "dataset": {
                "imageNetA": True,
                "naug": 8,
                "aug_type": "identity",
            },
            "drop": 0.5,
            "run": drop_tests and (DATASET_TO_TEST in ["a", "both"]),
        },
        "DROP ImageNetV2, without topk selection": {
            "memo": {
                "top": 1,
                "mapping": mapping_v2,
                "ensemble": True,
            },
            "dataset": {
                "imageNetA": False,
                "naug": 8,
                "aug_type": "identity",
            },
            "drop": 0.5,
            "run": drop_tests and (DATASET_TO_TEST in ["v2", "both"]),
        },
        "DROP ImageNetA, with topk selection": {
            "memo": {
                "top": 0.1,
                "mapping": mapping_a,
                "ensemble": True,
            },
            "dataset": {
                "imageNetA": True,
                "naug": 64,
                "aug_type": "identity",
            },
            "drop": 0.5,
            "run": drop_tests and (DATASET_TO_TEST in ["a", "both"]),
        },
        "DROP ImageNetV2, with topk selection": {
            "memo": {
                "top": 0.1,
                "mapping": mapping_v2,
                "ensemble": True,
            },
            "dataset": {
                "imageNetA": False,
                "naug": 64,
                "aug_type": "identity",
            },
            "drop": 0.5,
            "run": drop_tests and (DATASET_TO_TEST in ["v2", "both"]),
        },
        "DROP ImageNetA, cut ensemble without topk selection": {
            "memo": {
                "top": 1,
                "mapping": mapping_a,
                "ensemble": True,
            },
            "dataset": {
                "imageNetA": True,
                "naug": 8,
                "aug_type": "cut",
            },
            "drop": 0,
            "run": ensemble_tests and (DATASET_TO_TEST in ["a", "both"]),
        },
        "DROP ImageNetV2, cut ensemble  without topk selection": {
            "memo": {
                "top": 1,
                "mapping": mapping_v2,
                "ensemble": True,
            },
            "dataset": {
                "imageNetA": False,
                "naug": 8,
                "aug_type": "identity",
            },
            "drop": 0,
            "run": ensemble_tests and (DATASET_TO_TEST in ["v2", "both"]),
        },
        "DROP ImageNetA, cut ensemble  with topk selection": {
            "memo": {
                "top": 0.1,
                "mapping": mapping_a,
                "ensemble": True,
            },
            "dataset": {
                "imageNetA": True,
                "naug": 64,
                "aug_type": "identity",
            },
            "drop": 0,
            "run": ensemble_tests and (DATASET_TO_TEST in ["a", "both"]),
        },
        "DROP ImageNetV2, cut ensemble  with topk selection": {
            "memo": {
                "top": 0.1,
                "mapping": mapping_v2,
                "ensemble": True,
            },
            "dataset": {
                "imageNetA": False,
                "naug": 64,
                "aug_type": "identity",
            },
            "drop": 0,
            "run": ensemble_tests and (DATASET_TO_TEST in ["v2", "both"]),
        },
    }

    for t in tests:
        if tests[t]["run"]:
            arg = base_test | tests[t]
            arg['memo'] = base_test['memo'] | tests[t]['memo']
            arg['dataset'] = base_test['dataset'] | tests[t]['dataset']
            print(arg['memo']['device'])
            MEMO_testing_step(t, arg)
