import sys
sys.path.append('.')
from memo.test_calls.test_adapt import test_adapt

if __name__ == "__main__":
    model_name = 'resnet'
    batch_size = 8
    lr = 0.00025 if model_name == 'resnet' else 0.0001
    weight_decay = 0 if model_name == 'resnet' else 0.01
    opt = 'SGD' if model_name == 'resnet' else 'adamw'
    niter = 1
    prior_strength = -1
    test_adapt(model_name, batch_size, lr, weight_decay, opt, niter, prior_strength)
