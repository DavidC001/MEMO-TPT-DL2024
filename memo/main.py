import sys
import torch
import time
import numpy as np
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from datetime import timedelta

sys.path.append('.')
from memo.utils import memo_get_datasets
from memo.models import EasyMemo


def MEMO_testing_step(model, dataset, mapping: bool | list, test):
    print(f"Starting {test} evaluation...")
    start = time.time()

    np.random.seed(0)
    torch.manual_seed(0)

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

    imageNet_A, imageNet_V2 = memo_get_datasets('augmix', 1)
    mapping_a = [int(x) for x in imageNet_A.classnames.keys()]
    mapping_v2 = [int(x) for x in imageNet_V2.classnames.keys()]
    net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    memo_tests = True
    drop_tests = True
    ensemble_tests = True
    baseline_tests = False

    # Baseline tests
    if baseline_tests:
        memo = EasyMemo(net.to(device), device, mapping_a, prior_strength=1, top=1, ensemble=True)
        test_name = "Baseline ImageNetA"
        MEMO_testing_step(memo, imageNet_A, mapping_a, test_name)

        test_name = "Baseline ImageNetV2"
        del memo
        memo = EasyMemo(net.to(device), device, mapping_v2, prior_strength=1, top=1, ensemble=True)
        MEMO_testing_step(memo, imageNet_V2, mapping_v2, test_name)

        test_name = "ResNet50 V1 weights ImageNetA"
        del memo
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        memo = EasyMemo(net.to(device), device, mapping_a, prior_strength=1, top=1, ensemble=True)
        MEMO_testing_step(memo, imageNet_A, mapping_a, test_name)

        test_name = "ResNet50 V1 weights ImageNetV2"
        del memo
        memo = EasyMemo(net.to(device), device, mapping_v2, prior_strength=1, top=1, ensemble=True)
        MEMO_testing_step(memo, imageNet_V2, mapping_v2, test_name)
    del net, imageNet_A, imageNet_V2
    net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    imageNet_A, imageNet_V2 = memo_get_datasets('cut', 8)
    if memo_tests:
        test_name = "MEMO ImageNetA, without topk selection"
        memo = EasyMemo(net.to(device), device, mapping_a, prior_strength=0.94, top=1)
        MEMO_testing_step(memo, imageNet_A, mapping_a, test_name)

        test_name = "MEMO ImageNetV2, without topk selection"
        del memo
        memo = EasyMemo(net.to(device), device, mapping_v2, prior_strength=0.94, top=1)
        MEMO_testing_step(memo, imageNet_V2, mapping_v2, test_name)

        test_name = "MEMO ImageNetA, with topk selection"
        del memo, imageNet_V2, imageNet_A
        imageNet_A, imageNet_V2 = memo_get_datasets('cut', 64)
        memo = EasyMemo(net.to(device), device, mapping_a, prior_strength=0.94, top=0.1)
        MEMO_testing_step(memo, imageNet_A, mapping_a, test_name)

        test_name = "MEMO ImageNetV2, with topk selection"
        del memo
        memo = EasyMemo(net.to(device), device, mapping_v2, prior_strength=0.94, top=0.1)
        MEMO_testing_step(memo, imageNet_V2, mapping_v2, test_name)

    # Dropout tests
    if drop_tests:
        test_name = "DROP ImageNetA, without topk selection"
        net.layer4.add_module('dropout', nn.Dropout(0.5, inplace=True))
        del memo, imageNet_V2, imageNet_A
        imageNet_A, imageNet_V2 = memo_get_datasets('identity', 8)
        memo = EasyMemo(net.to(device), device, mapping_a, prior_strength=1, top=1, ensemble=True)
        MEMO_testing_step(memo, imageNet_A, mapping_a, test_name)

        test_name = "DROP ImageNetV2, without topk selection"
        del memo
        memo = EasyMemo(net.to(device), device, mapping_v2, prior_strength=1, top=1, ensemble=True)
        MEMO_testing_step(memo, imageNet_V2, mapping_v2, test_name)

        test_name = "DROP ImageNetA, with topk selection"
        del memo, imageNet_V2, imageNet_A
        imageNet_A, imageNet_V2 = memo_get_datasets('identity', 64)
        memo = EasyMemo(net.to(device), device, mapping_a, prior_strength=1, top=0.1, ensemble=True)
        MEMO_testing_step(memo, imageNet_A, mapping_a, test_name)

        test_name = "DROP ImageNetV2, with topk selection"
        del memo
        memo = EasyMemo(net.to(device), device, mapping_v2, prior_strength=1, top=0.1, ensemble=True)
        MEMO_testing_step(memo, imageNet_V2, mapping_v2, test_name)

    if ensemble_tests:
        test_name = "DROP ImageNetA, cut ensemble without topk selection"
        del memo, imageNet_V2, imageNet_A, net
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        net.layer4.add_module('dropout', nn.Dropout(0, inplace=True))
        imageNet_A, imageNet_V2 = memo_get_datasets('cut', 8)
        memo = EasyMemo(net.to(device), device, mapping_a, prior_strength=1, top=1, ensemble=True)
        MEMO_testing_step(memo, imageNet_A, mapping_a, test_name)

        test_name = "DROP ImageNetV2, cut ensemble  without topk selection"
        del memo
        memo = EasyMemo(net.to(device), device, mapping_v2, prior_strength=1, top=1, ensemble=True)
        MEMO_testing_step(memo, imageNet_V2, mapping_v2, test_name)

        test_name = "DROP ImageNetA, cut ensemble  with topk selection"
        del memo, imageNet_V2, imageNet_A
        imageNet_A, imageNet_V2 = memo_get_datasets('cut', 64)
        memo = EasyMemo(net.to(device), device, mapping_a, prior_strength=1, top=0.1, ensemble=True)
        MEMO_testing_step(memo, imageNet_A, mapping_a, test_name)

        test_name = "DROP ImageNetV2, cut ensemble  with topk selection"
        del memo
        memo = EasyMemo(net.to(device), device, mapping_v2, prior_strength=1, top=0.1, ensemble=True)
        MEMO_testing_step(memo, imageNet_V2, mapping_v2, test_name)
