import sys
import argparse
sys.path.append(".")

import torch
import numpy as np

from Ensemble.functions import runTest


def main():
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run Ensemble tests",
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
        "-v",
        "--verbose",
        type=int,
        help="Frequency of verbose output",
        default=100,
        metavar="",
    )

    args = vars(parser.parse_args())
    DATASET_ROOT = args["datasets_root"]
    Tests = {
        "ImageNet-A RN50 + RNXT": {
            "imageNetA": True,
            "naug": 64,
            "top": 0.1,
            "niter": 1,
            "testSingleModels": True,
            "simple_ensemble": True,
            "device": "cuda",

            "models_type": ["memo", "memo"],
            "args": [
                {"device": "cuda", "drop": 0, "ttt_steps": 1, "model": "RN50"},
                {"device": "cuda", "drop": 0, "ttt_steps": 1, "model": "RNXT"}
            ],
            "temps": [1, 1],
            "names": ["MEMO RN50", "MEMO RNXT"],
            "dataset_root": DATASET_ROOT,
        },

        "ImageNet-V2 TPT RN50 + RNXT": {
            "imageNetA": False,
            "naug": 64,
            "top": 0.1,
            "niter": 1,
            "testSingleModels": True,
            "simple_ensemble": True,
            "device": "cuda",

            "models_type": ["memo", "memo"],
            "args": [
                {"device": "cuda", "drop": 0, "ttt_steps": 1, "model": "RN50"},
                {"device": "cuda", "drop": 0, "ttt_steps": 1, "model": "RNXT"}
            ],
            "temps": [1, 1],
            "names": ["MEMO RN50", "MEMO RNXT"],
            "dataset_root": DATASET_ROOT,
        },

        "ImageNet-A TPT + MEMO": {
            "imageNetA": True,
            "naug": 64,
            "top": 0.1,
            "niter": 1,
            "testSingleModels": True,
            "simple_ensemble": True,
            "device": "cuda",

            "models_type": ["memo", "tpt"],
            "args": [
                {"device": "cuda", "drop": 0, "ttt_steps": 1, "model": "RN50"},
                {"device": "cuda", "ttt_steps": 1, "align_steps": 0, "arch": "RN50"}
            ],
            "temps": [1.55, 0.7],
            "names": ["MEMO", "TPT"],
            "dataset_root": DATASET_ROOT,
        },

        "ImageNet-A RN50 + RNXT + TPT": {
            "imageNetA": True,
            "naug": 64,
            "top": 0.1,
            "niter": 1,
            "testSingleModels": True,
            "simple_ensemble": True,
            "device": "cuda",

            "models_type": ["memo", "memo", "tpt"],
            "args": [
                {"device": "cuda", "drop": 0, "ttt_steps": 1, "model": "RN50"},
                {"device": "cuda", "drop": 0, "ttt_steps": 1, "model": "RNXT"},
                {"device": "cuda", "ttt_steps": 1, "align_steps": 0, "arch": "RN50"}
            ],
            "temps": [1, 1, 0.7],
            "names": ["MEMO-RN50", "MEMO-RNXT", "TPT"],
            "dataset_root": DATASET_ROOT,
        },
    }

    for test in Tests:
        print(f"Running test: {test}")
        test = Tests[test]
        test["verbose"] = args["verbose"]
        result = runTest(**test)

        print("\tFInal Results:")
        for key in result:
            print(f"\t\t{key}: {result[key]}")

        print("\n-------------------\n")


if __name__ == "__main__":
    main()
