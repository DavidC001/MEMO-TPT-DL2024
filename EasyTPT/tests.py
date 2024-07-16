import sys

sys.path.append(".")

import time
import torch
import wandb
import datetime
import numpy as np

from EasyTPT.utils import tpt_get_datasets
from EasyTPT.models import EasyTPT
from EasyTPT.setup import get_test_args

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    args = get_test_args()

    VERBOSE = args["verbose"]
    if VERBOSE == -1:
        VERBOSE = sys.maxsize
    DATA_TO_TEST = args["data_to_test"]
    DATASET_ROOT = args["datasets_root"]
    WANDB_SECRET = args["wandb_secret"]

    if WANDB_SECRET != "":
        wandb.login(key=WANDB_SECRET)

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

    # test_stop stops the testing after a certain number of samples
    # to run the entire dataset keep it at -1
    tests = []
    if DATA_TO_TEST in ["a", "both"]:
        print("[TEST] Running tests on ImageNet A")
        tests = [
            {
                "name": "TPT_sel_A",
                "dataset": "A",
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
                "name": "TPT_align_A",
                "dataset": "A",
                "align_steps": 1,
            },
        ]

    if DATA_TO_TEST in ["v2", "both"]:
        print("[TEST] Running tests on ImageNet V2")
        tests = (tests if DATA_TO_TEST == "both" else []) + [
            {
                "name": "TPT_sel_V2",
                "dataset": "V2",
                "augs": 32,
                "confidence": 0.2,
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
                "augs": 64,
                "ensemble": True,
                "confidence": 0.1,
            },
            {
                "name": "TPT_align_sel_V2",
                "dataset": "V2",
                "augs": 32,
                "confidence": 0.2,
                "align_steps": 1,
            },
        ]

    for idx, settings in enumerate(tests):

        test = base_test | settings

        dataset_name = test["dataset"]
        test_name = test["name"]
        device = test["device"]

        BASE_PROMPT = test["base_prompt"]
        ARCH = test["arch"]
        SPLT_CTX = test["splt_ctx"]
        LR = test["lr"]
        AUGS = test["augs"]
        TTT_STEPS = test["ttt_steps"]
        ALIGN_STEPS = test["align_steps"]
        ENSEMBLE = test["ensemble"]
        TEST_STOP = test["test_stop"]
        CONFIDENCE = test["confidence"]

        if WANDB_SECRET != "":
            timestamp = time.strftime("%m%d%H%M%S")
            run_name = f"{test_name}_{timestamp}"
            wandb.init(
                project="MEMOTPT",
                name=run_name,
                config={
                    "test_name": test_name,
                    "dataset_name": dataset_name,
                    "base_prompt": BASE_PROMPT,
                    "arch": ARCH,
                    "splt_ctx": str(SPLT_CTX),
                    "lr": LR,
                    "augs": AUGS,
                    "ttt_steps": TTT_STEPS,
                    "align_steps": ALIGN_STEPS,
                    "ensemble": ENSEMBLE,
                    "test_stop": TEST_STOP,
                    "confidence": CONFIDENCE,
                },
            )

        print("-" * 30)
        print(f"[TEST] Running test {idx + 1} of {len(tests)}: {test_name} \n{test}")

        print(f"[TEST] loading datasets with {AUGS} augmentation...")
        datasetRoot = DATASET_ROOT
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
            del imageNetV2, imageNetV2CustomNames, imageNetV2Map
        elif dataset_name == "V2":
            print("[TEST] using ImageNet V2")
            dataset = imageNetV2
            classnames = imageNetV2CustomNames
            id_mapping = imageNetV2Map
            del imageNetA, imageNetACustomNames, imageNetAMap

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
        total_time = 0

        idxs = [i for i in range(len(dataset))]

        SEED = 1
        np.random.seed(SEED)
        np.random.shuffle(idxs)

        for idx in idxs:
            data = dataset[idx]
            label = data["label"]
            imgs = data["img"]
            name = data["name"]

            start = time.time()

            cnt += 1
            with torch.no_grad():
                tpt.reset()

            out_id = tpt.predict(imgs)
            tpt_predicted = classnames[out_id]

            end = time.time()

            total_time += end - start
            avg_time = total_time / cnt

            if int(id_mapping[out_id]) == label:
                emoji = ":D"
                tpt_correct += 1
            else:
                emoji = ":("

            tpt_acc = tpt_correct / (cnt)

            if WANDB_SECRET != "":
                wandb.log({"tpt_acc": tpt_acc})

            if cnt % VERBOSE == 0:
                print(emoji)
                print(f"TPT Accuracy: {round(tpt_acc, 3)}")
                print(f"GT: \t{name}\nTPT: \t{tpt_predicted}")
                print(
                    f"after {cnt} samples, average time {round(avg_time, 3)}s ({round(1 / avg_time, 3)}it/s)\n"
                )

            if cnt == TEST_STOP:
                print(f"[TEST] Early stopping at {cnt} samples")
                break

        del tpt

        print(f"[TEST] Final TPT Accuracy: {round(tpt_acc, 3)} over {cnt} samples")

        if WANDB_SECRET != "":
            wandb.finish()
