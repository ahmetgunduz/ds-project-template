# Repository Design <!-- omit in toc -->

This document contains a step-by-step walkthrough of the design of the repository.

## Table of Contents <!-- omit in toc -->

- [1. Requirements](#1-requirements)
- [2. Design decisions](#2-design-decisions)
- [3. Directory structure](#3-directory-structure)

## 1. Requirements


All design starts from requirements. In the case of this project, The repository should be able to

- explore the dataset, and provide a way to download it.
- preprocess the dataset and track the versions of the data
- extract features
- train classification models and track experiments
- evaluate models
- predict on new data while ensuring the consistency in data preprocessing and feature extraction steps
- reproduce
- collaborate
- consistent code style

## 2. Design decisions

The requirements stated above naturally lead to a number of design decisions.
Below the requirements are again listed, accompanied by the design decisions made for each of them:

- the ability to explore the dataset, and provide a way to download it.
    - the exploration should be done in a notebook as it is a very interactive process. `notebooks/01-exploratory-data-analysis.ipynb` is the notebook for this purpose.
        - It is important to note that the notebook is not meant to be run in a single go. It is meant to be run in a step-by-step fashion, with the user exploring the dataset and making decisions on how to proceed.
    - The dataset is stored in Google Drive, and can be downloaded using the `gdown` package using the `src/data/create_dataset.py` script.
    - furthermore, quick-and-dirty experimentation is facilitated by having a `notebooks` directory, where there are no requirements for the code functionality or style

- the ability to preprocess the dataset and track the versions of the data

    - versioning is achieved using a folder structure, where each version of the dataset is stored in a separate folder. The folder structure is described in the [Directory structure](#directory-structure) section.
    - after downloading the dataset from Google Drive and preprocessng, the `src/data/create_dataset.py` script is used.
- the ability to extract features
    - the features are extracted using the `src/features/build_features.py` script.
    - the features are stored in a separate folder `data/features`.

- the ability to train classification models and track experiments
    - the models are trained using the `src/models/train_model.py` script.
    - the models are stored in a separate folder.
    - the experiments are tracked using [Weights and Biases (wandb)](https://wandb.ai/home)

- the ability to evaluate models
    - Since we do not have any labeled validation set in the dataset, we used cross validation to evaluate the models. The evaluation is done during the training process, and the results are stored in the `wandb` dashboard.

- the ability to predict on new data while ensuring the consistency in data preprocessing and feature extraction steps
    - the predictions are done using the `src/models/predict_model.py` script.
    - consistency is ensured by using pipeline objects from `sklearn` to store the preprocessing and feature extraction steps. In this project one pipeline is constructed using `sklearn` so we stored in the `outputs/models` folder. If we had multiple pipelines, we would have stored them in a separate folder like `outputs/pipelines`.

- the ability to reproduce
    - in terms of code structure and style, this is achieved by mandating that every script (data acquisition, preprocessing, dataset composition, model training, model prediciton) cleanly separates configuration from code, and provides a means to pass this configuration to the code
    - this configuration is then stored in a file
    - variations on the configuration can then be stored as files with a different name
    - furthermore, it facilitates the practice of versioning: aspects of the code or configuration can evolve, while older versions remain available and reproducible
    - in terms of code, this is facilitated by the usage of a version control system (`git`)
    - in terms of data, this is facilitated by having the data preprocessing and dataset composition scripts be deterministic based on configuration
    - for Python scripts, [Typer](https://typer.tiangolo.com/) is a good choice for passing parameters to a script

- the ability to collaborate
  - in terms of tooling, this is achieved through the usage of `git`, experiment tracking software, and potential other tools (e.g. Google Drive)
  - in terms of code, this is achieved by using tools that enforce a consistent style and structure (e.g. `black` for code formatting)
  - in terms of models, this is achieved by saving trained models to a place where everyone can access them
  - more generally, this is facilitated by having good documentation, which was a prime reason to write the documentation you are currently reading
  - pre-commit hooks are used to ensure that the code is formatted and linted before committing

- consistent code style
  - there are many tools in this category, and the choice will be mostly a matter of personal preference
  - regarding the code style, I have chosen to use [black](https://black.readthedocs.io/en/stable/), combined with [isort](https://pycqa.github.io/isort/), [flake8](https://flake8.readthedocs.io/en/latest/), and [mypy](https://mypy.readthedocs.io/en/stable/)
  - for enforcing these tools, there is an excellent tool called [pre-commit](https://pre-commit.ci/), which will automatically run scripts defined in `.pre-commit-config.yaml` before every commit
  - it will also run other useful checks, such as checking for commited files above a certain size, formatting json, and so on

## 3. Directory structure

The design decisions above lead to a certain directory structure. The projects general structure  is created by `cookiecutter` libraries data science project: https://github.com/drivendata/cookiecutter-data-science . But we have made some changes to the structure to fit our needs.
Project structure is as follows:

```
├── LICENSE
├── README.md                   <- The top-level README for developers using this project.
├── data
│   ├── external                <- Data from third party sources.
│   ├── feature                 <- Intermediate data that has been transformed.
│   ├── processed               <- The final, canonical data sets for modeling.
│   └── raw                     <- The original, immutable data dump.
├── docs                        <- Documentation files
├── outputs                     <- Trained and serialized models, model predictions, or model summaries
│   ├── models                  <- Trained models
│   └── predictions             <- Model predictions
├── notebooks                   <- Jupyter notebooks. Naming convention is a number (for ordering),
│                                  the creator's initials, and a short `-` delimited description, e.g.
│                                  `01-job-name`.
├── reports                     <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                 <- Generated graphics and figures to be used in reporting
├── src                         <- Source code for use in this project.
│   ├── __init__.py             <- Makes src a Python module
│   ├── data                    <- Scripts to download or generate data
│   │   └── create_dataset.py
│   ├── features                <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py   <- Feature engineering
│   ├── models                  <- Scripts to train models and then use trained models to make
│   │                               predictions
│   │   ├── predict_model.py    <- Make predictions on new data
│   │   └── train_model.py      <- Train and evaluate model on training data
│   └── visualization           <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py        <- Create exploratory and results oriented visualizations
|   ├── utils/                  <- Utility functions
│   │   └── experiment_tracking.py
│   ├── enums.py                <- Enums
│   ├── logger.py               <- Logger
│   └── paths.py                <- Paths
├── tests                       <- Unit tests
├── outputs                     <- Trained and serialized models, model predictions, or model summaries
├── vars.env                    <- Environment variables
├── secrets.env                 <- Secret environment variables (not added to git)
├── .pre-commit-config.yaml     <- pre-commit hooks
├── .gitignore                  <- gitignore file
├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
│                                  generated with `pipreqs . --force`
├── conda.yaml                  <- The requirements file for reproducing the analysis environment, e.g.
│                                  generated with `conda env export --no-builds > conda.yaml`
└── pyproject.toml               <- Configuration file for black and isort
```
