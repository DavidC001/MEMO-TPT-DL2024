# MEMO-TPT-DL2024
![image](https://github.com/user-attachments/assets/63853bb5-9746-4020-baf8-58bad44aaf15)

This repository contains the project for the master's course in Deep Learning, focusing on Test Time Robustness via Adaptation and Augmentation (MEMO) and Test-Time Prompt Tuning (TPT).

## How to Use

### Prerequisites
Ensure you have the required dependencies by installing them via `requirements.txt`:

```bash
pip install -r requirements.txt
```

And have the ImageNetA and ImageNetV2 test datasets available. To have the scripts work "out of the box" follow this directory structure:

```bash
MEMO-TPT-DL2024
├── datasets
│   ├── imagenet-a
│   │   └── ...
│   └── imagenetv2-matched-frequency-format-val
│       └── ...
└── ...
```

### Running the Project

1. **Setup the Data Loaders**
   - Configure the `data_loader.py` to fetch ImageNetA and ImageNetV2 datasets.

2. **Running on AWS**
   - Use the `DL_AWS_project.ipynb` notebook to run the project on AWS. This notebook is self-contained and includes all necessary steps to initialize and run the models.<br>
   NOTE: This needs to have the datasets uploaded to an S3 bucket and the necessary configurations set up.

3. **Training and Testing single Models**

   - **EasyTPT Module**:
     - Navigate to the `EasyTPT` directory.
     - Run `tests.py` to train and test different configurations of the EasyTPT model.

   - **MEMO Module**:
     - Navigate to the `MEMO` directory.
     - Run `main.py` to train and test different configurations of the EasyMemo model.

   - **Ensemble Module**:
     - Navigate to the `Ensemble` directory.
     - Initialize and run the ensemble tests by executing `main.py`.

### Directory Structure

```
MEMO-TPT-DL2024
├── README.md
├── .gitignore
├── requirements.txt
|
├── EasyModel.py
├── DL_AWS_project.ipynb - self-contained notebook to run the project on AWS
|
├── dataloaders
│   ├── data_loader.py - get ImageNetA and ImageNetV2 datasets
│   ├── easyAugmenter.py - EasyAugmenter class
│   ├── imageNetA.py - ImageNetA dataset class
│   ├── imageNetV2.py - ImageNetV2 dataset class
│   └── wordNetIDs2Classes.csv - mapping of WordNet IDs to classes
│
├── EasyTPT
│   ├── IMPLEMENTATION.md
│   ├── models.py 
│   ├── utils.py 
│   ├── setup.py 
│   ├── main.py
│   ├── tests.py
│   └── ...
│
├── MEMO
│   ├── IMPLEMENTATION.md
│   ├── models.py
│   ├── utils.py
│   └── main.py
|
└── Ensemble
    ├── IMPLEMENTATION.md
    ├── models.py
    ├── utils.py
    ├── functions.py
    └── main.py
```
# Models
![Diagrammi UML Deep Learning - Page 1](https://github.com/DavidC001/MEMO-TPT-DL2024/assets/40665241/09461091-b12b-4379-88ff-8530e04e1255)


## TPT
The TPT (Test-Time Prompt Tuning) module, represented by the `EasyTPT` class, is designed to use CLIP with prompt tuning. This class incorporates various configurations for model architecture, prompt settings, and optimization parameters. Key functionalities include:
- Initialization with model architecture, device setup, and prompt configurations.
- Methods for model forward pass, inference, and prediction.
- Optimizer setup and selection of confident samples.

### Usage

There are two ways to run EasyTPT in its different configurations:

```
python EasyTPT/tests.py [-h] [-v] [--data-to-test] [-d]

Run TPT tests

options:
  -h, --help            show this help message and exit
  -v , --verbose        Frequency of verbose output
  -d , --datasets-root  Root folder of all the datasets, default='datasets'
  --data-to-test        Which dataset to test between 'a', 'v2', 'both'
```

`test.py` contains a set of predefined tests that can be run to evaluate the model's performance on ImageNetA, ImageNetV2, or both datasets. It's possible to edit or define custom tests by directly modifying the *tests* and *tpt_base_test* lists in the script. In each test it's possible to define an additional *test_stop* parameter in order to stop the test after a certain number of iterations.

```
python EasyTPT/main.py [-h] [-ar] [-p] [-t] [-au] [-S] [-A] [-as] [-en]

Run TPT

options:
  -h, --help            show this help message and exit
  -ar , --arch          Architecture to use
  -p , --base-prompt    Base prompt to use
  -t , --tts            Number of tts
  -au , --augs          Number of augmentations (includes original image)
  -S, --single-context  Split context or not
  -A, --augmix          Use AugMix or not
  -as , --align_steps   Number of alignment steps
  -en, --ensemble       Use ensemble mode
```

`main.py` allows to run TPT with custom configurations, each parameter defined via command line. By default the script will run the model on ImageNetA but ImageNetV2 is aviable too.


## MEMO
The MEMO (Meta Ensemble Model Optimization) module, represented by the `EasyMemo` class, focuses on optimizing model ensembles through meta-learning techniques. This class includes:
- Initialization with model parameters, device setup, and optimization configurations.
- Methods for model forward pass, inference, and prediction.
- Optimizer setup and selection of confident samples.

## Ensemble
The `Ensemble` class aggregates multiple `EasyModel` instances to enhance prediction accuracy and reliability. It provides:
- Initialization with a list of models and temperature settings for the ensemble.
- Ensemble forward pass and reset functionality.

### Usage
To test the ensamble of models, use the script `main.py` under the `Ensemble` directory. The script allows to define the models to be used in the ensemble and the temperature to be applied to the predictions, an example of a valid experiment definition is the following:

```python
{
    "TEST_NAME": {
        "imageNetA" : True, # if the test should be performed on ImageNet-A or on ImageNet-V2
        "naug" : 64, # number of augmentations to perform
        "top" : 0.1, # percentage of the top augmentations to consider (confidence selection)
        "niter" : 1, # number of optimization steps
        "testSingleModels" : True, # if we wanto to also compute the results for the single models
        "simple_ensemble" : True, # if we want to also compute the results for the simple ensemble strategy
        "device" : "cuda", # device to use for the computation
            
        "models_type" : ["memo", "tpt", "..."], # list of models types to use for the ensemble, can be "memo" or "tpt"
        "args" : [ # arguments for each model
            {"device": "cuda", "drop": 0, "ttt_steps": 1, "model": "RN50"}, # arguments for the first model
            {"device": "cuda", "ttt_steps": 1, "align_steps": 0, "arch": "RN50"}, # arguments for the second model
            "..."
            ],
        "temps" : [1.55, 0.7], # temperature rescaling to use for each model
        "names" : ["MEMO", "TPT"], # names to use for each model
    }
}
```

To execute the script, run the following command:

```bash
python main.py

optional arguments:
  -v, --verbose         Frequency of verbose output
  -d , --datasets-root  Root folder of all the datasets, default='datasets'
```

# Report
The final project report notebook is available in the `DL_AWS_project.ipynb` file. This notebook includes detailed explanations of the project, implementation details, results, and discussions.
It can be run on AWS with the necessary configurations and datasets to reproduce the results.
