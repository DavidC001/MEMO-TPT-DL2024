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
        "-C",
        "--clip",
        action="store_true",
        help="Run parallel with CLIP or not",
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

    args = vars(parser.parse_args())
    return args
