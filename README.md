# MEMO-TPT-DL2024

This repository contains the project for the master's course in Deep Learning, focusing on Meta Ensemble Model Optimization (MEMO) and Transformer Prompt Tuning (TPT).

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

## MEMO
The MEMO (Meta Ensemble Model Optimization) module, represented by the `EasyMemo` class, focuses on optimizing model ensembles through meta-learning techniques. This class includes:
- Initialization with model parameters, device setup, and optimization configurations.
- Methods for model forward pass, inference, and prediction.
- Optimizer setup and selection of confident samples.

## Ensemble
The `Ensemble` class aggregates multiple `EasyModel` instances to enhance prediction accuracy and reliability. It provides:
- Initialization with a list of models and temperature settings for the ensemble.
- Ensemble forward pass and reset functionality.

# Report
The final project report notebook is available in the `DL_AWS_project.ipynb` file. This notebook includes detailed explanations of the project, implementation details, results, and discussions.
It can be run on AWS with the necessary configurations and datasets to reproduce the results.