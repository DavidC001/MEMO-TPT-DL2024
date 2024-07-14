import argparse


def get_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run TPT",
    )

    parser.add_argument(
        "-ar",
        "--arch",
        type=str,
        help="Architecture to use",
        default="RN50",
        metavar="",
    )

    parser.add_argument(
        "-p",
        "--base-prompt",
        type=str,
        help="Base prompt to use",
        default="a photo of a [CLS]",
        metavar="",
    )

    parser.add_argument(
        "-t",
        "--tts",
        type=int,
        help="Number of tts",
        default=1,
        metavar="",
    )

    parser.add_argument(
        "-au",
        "--augs",
        type=int,
        help="Number of augmentations (includes original image)",
        default=64,
        metavar="",
    )

    parser.add_argument(
        "-S",
        "--single-context",
        action="store_true",
        help="Split context or not",
        default=False,
    )

    parser.add_argument(
        "-A",
        "--augmix",
        action="store_true",
        help="Use AugMix or not",
        default=False,
    )

    parser.add_argument(
        "-as",
        "--align_steps",
        type=int,
        help="Number of alignment steps",
        default=0,
        metavar="",
    )

    parser.add_argument(
        "-en",
        "--ensemble",
        action="store_true",
        help="Use ensemble mode",
        default=False,
    )

    args = vars(parser.parse_args())
    return args


def get_test_args():
    parser = argparse.ArgumentParser(
        prog="tests.py",
        description="Run TPT tests",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        help="Frequency of verbose output",
        default=100,
        metavar="",
    )

    parser.add_argument(
        "--data-to-test",
        type=str,
        help="Which dataset to test between 'a', 'v2', 'both'",
        default="both",
        metavar="",
    )

    parser.add_argument(
        "-d",
        "--datasets-root",
        type=str,
        help="Root folder of all the datasets, default='datasets'",
        default="datasets",
        metavar="",
    )

    parser.add_argument(
        "-w",
        "--wandb-secret",
        type=str,
        help="Wandb secret key",
        default="",
        metavar="",
    )

    args = vars(parser.parse_args())
    return args
