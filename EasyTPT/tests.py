import sys

sys.path.append(".")

import torch
import numpy as np

from EasyTPT.utils import tpt_get_datasets
from EasyTPT.models import EasyTPT


if __name__ == "__main__":

    VERBOSE = True

    base_test = {
        "name": "Base",
        "dataset": "A",
        "augs": 64,
        "ttt_steps": 1,
        "align_steps": 0,
        "ensemble": False,
        "test_stop": -1,
        "confidence": 0.10,
        "base_prompt": "A photo of a [CLS]",
        "arch": "RN50",
        "splt_ctx": True,
        "lr": 0.005,
        "device": "cuda:0",
    }

    # test_step stops the testing after a certain number of samples
    # to run the entire dataset keep it at -1
    tests = [
        {
            "name": "TPT_sel_A",
            "dataset": "A",
        },
        {
            "name": "TPT_sel_V2",
            "dataset": "V2",
        },
        {
            "name": "TPT_ens_nosel_A",
            "dataset": "A",
            "augs": 8,
            "ensemble": True,
            "confidence": 1,
        },
        {
            "name": "TPT_ens_sel_A",
            "dataset": "A",
            "ensemble": True,
            "confidence": 0.10,
        },
        {
            "name": "TPT_ens_nosel_V2",
            "dataset": "V2",
            "augs": 8,
            "ensemble": True,
            "confidence": 1,
        },
        {
            "name": "TPT_ens_sel_V2",
            "dataset": "V2",
            "ensemble": True,
            "confidence": 0.10,
        },
        {
            "name": "TPT_align_A",
            "dataset": "A",
            "align_steps": 1,
        },
        {
            "name": "TPT_align_V2",
            "dataset": "V2",
            "align_steps": 1,
        },
    ]

    for settings in tests:

        test = base_test | settings

        dataset_name = test["dataset"]
        test_name = test["name"]
        device = test["device"]

        BASE_PROMPT = test["base_prompt"]
        ARCH = test["arch"]
        SPLT_CTX = test
        LR = test["lr"]
        AUGS = test["augs"]
        TTT_STEPS = test["ttt_steps"]
        ALIGN_STEPS = test["align_steps"]
        ENSEMBLE = test["ensemble"]
        TEST_STOP = test["test_stop"]
        CONFIDENCE = test["confidence"]

        print("-" * 30)
        print(f"[TEST] Running test {test_name}: {test}")

        print(f"[TEST] loading datasets with {AUGS} augmentation...")
        datasetRoot = "datasets"
        (
            imageNetA,
            _,
            imageNetACustomNames,
            imageNetAMap,
            imageNetV2,
            _,
            imageNetV2CustomNames,
            imageNetV2Map,
        ) = tpt_get_datasets(datasetRoot, augs=AUGS, all_classes=False)
        print("[TEST] datasets loaded.")

        if dataset_name == "A":
            print("[TEST] using ImageNet A")
            dataset = imageNetA
            classnames = imageNetACustomNames
            id_mapping = imageNetAMap
        elif dataset_name == "V2":
            print("[TEST] using ImageNet V2")
            dataset = imageNetV2
            classnames = imageNetV2CustomNames
            id_mapping = imageNetV2Map

        tpt = EasyTPT(
            device,
            base_prompt=BASE_PROMPT,
            arch=ARCH,
            splt_ctx=SPLT_CTX,
            classnames=classnames,
            ttt_steps=TTT_STEPS,
            lr=LR,
            align_steps=ALIGN_STEPS,
            ensemble=ENSEMBLE,
            confidence=CONFIDENCE,
        )

        cnt = 0
        tpt_correct = 0

        idxs = [i for i in range(len(dataset))]

        SEED = 1
        np.random.seed(SEED)
        np.random.shuffle(idxs)

        for idx in idxs:
            data = dataset[idx]
            label = data["label"]
            imgs = data["img"]
            name = data["name"]

            with torch.no_grad():
                tpt.reset()

            if ALIGN_STEPS > 0:
                if VERBOSE:
                    print(f"[EasyTPT] Aligning embeddings for {ALIGN_STEPS} steps")
                tpt.align_embeddings(imgs)

            out_id = tpt.predict(imgs)
            tpt_predicted = classnames[out_id]

            if int(id_mapping[out_id]) == label:
                if VERBOSE:
                    print(":D")
                tpt_correct += 1
            else:
                if VERBOSE:
                    print(":(")
            cnt += 1

            tpt_acc = tpt_correct / (cnt)

            if VERBOSE:
                print(f"TPT Accuracy: {round(tpt_acc,3)}")
                print(f"GT: \t{name}\nTPT: \t{tpt_predicted}")
                print(f"after {cnt} samples\n")

            if cnt == TEST_STOP:
                print(f"[TEST] Early stopping at {cnt} samples")
                break

        print(f"[TEST] Final TPT Accuracy: {round(tpt_acc,3)} over {cnt} samples")


breakpoint()
