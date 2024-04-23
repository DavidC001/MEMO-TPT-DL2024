import sys
sys.path.append('.')
from memo.test_calls.test_adapt import test_adapt

dataroot = '/path/to/imagenet/datasets/'  # EDIT THIS

experiments = {0: 'imagenetA', 1: 'imagenetV2'}

if __name__ == "__main__":
    test_adapt()
