import sys
sys.path.append(".")

import torch
import numpy as np

from Ensemble.functions import runTest

def main():
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)

    Tests = {
        "ImageNet-A RN50 + RNXT": {
            "imageNetA" : True,
            "naug" : 64,
            "top" : 0.1,
            "niter" : 1,
            "testSingleModels" : True,
            "simple_ensemble" : True,
            "device" : "cuda",
            
            "models_type" : ["memo", "memo"],
            "args" : [
                {"device": "cuda", "drop": 0, "ttt_steps": 1, "model": "RN50"},
                {"device": "cuda", "drop": 0, "ttt_steps": 1, "model": "RNXT"}
                ],
            "temps" : [1, 1],
            "names" : ["MEMO RN50", "MEMO RNXT"],
        },

        "ImageNet-V2 TPT RN50 + RNXT": {
            "imageNetA" : False,
            "naug" : 64,
            "top" : 0.1,
            "niter" : 1,
            "testSingleModels" : True,
            "simple_ensemble" : True,
            "device" : "cuda",
            
            "models_type" : ["memo", "memo"],
            "args" : [
                {"device": "cuda", "naug": 64, "drop": 0, "ttt_steps": 1, "model": "RN50"},
                {"device": "cuda", "naug": 64, "drop": 0, "ttt_steps": 1, "model": "RNXT"}
                ],
            "temps" : [1, 1],
            "names" : ["MEMO RN50", "MEMO RNXT"],
        },

        "ImageNet-A TPT + MEMO": {
            "imageNetA" : True,
            "naug" : 64,
            "top" : 0.1,
            "niter" : 1,
            "testSingleModels" : True,
            "simple_ensemble" : True,
            "device" : "cuda",
            
            "models_type" : ["memo", "tpt"],
            "args" : [
                {"device": "cuda", "drop": 0, "ttt_steps": 1, "model": "RN50"},
                {"device": "cuda", "ttt_steps": 1, "align_steps": 0, "arch": "RN50"}
                ],
            "temps" : [1.55, 0.7],
            "names" : ["MEMO", "TPT"],
        },
    }

    for test in Tests:
        print(f"Running test: {test}")
        test = Tests[test]
        runTest(**test)
        print("\n-------------------\n")
    

if __name__ == "__main__":
    main()