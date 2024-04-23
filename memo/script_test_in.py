from subprocess import call
from test_calls import test_adapt, test_initial

dataroot = '/path/to/imagenet/datasets/'  # EDIT THIS

experiments = {0: 'imagenetA', 1: 'imagenetV2'}

if resume in ('rvt', 'rn101', 'rn101_wsl'):
    model_tag = '--use_rvt' if resume == 'rvt' else '--use_resnext'
    optimizer = 'adamw'
    lr = 0.00001
    weight_decay = 0.01
else:
    model_tag = ''
    optimizer = 'sgd'
    lr = 0.00025
    weight_decay = 0.0

for corruption in ['adversarial']:
    for level in [0]:
        print(corruption, 'level', level)

        call(' '.join(['python', 'test_calls/test_adapt.py',
                       f'--dataroot {dataroot}',
                       model_tag,
                       f'--level {level}',
                       f'--corruption {corruption}',
                       f'--resume results/imagenet_{resume}/',
                       f'--optimizer {optimizer}',
                       f'--lr {lr}',
                       f'--weight_decay {weight_decay}']),
             shell=True)

if __name__ == "__main__":
    test_adapt()
