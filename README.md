# Catchjoe  <!-- omit in toc -->

A project template for Data Science projects with integration of Wandb, MLflow[TODO], and DVC [TODO].

## Table of Contents <!-- omit in toc -->

- [Documentation Links](#documentation-links)
- [Setup](#setup)
  - [Local](#local)
- [Usage](#usage)
  - [Data](#data)
  - [Features](#features)
  - [Models](#models)
  - [Predictions](#predictions)
- [Common issues](#common-issues)

## Documentation Links

Further documentation can be found in the `docs` directory.

- [Repository Design](docs/design.md)


## Setup

- clone the repository and `cd` into it

```sh
# using ssh
git clone <ssh git directory of the repo here>

# or using https
git clone <https git directory of the repo here>

# move terminal into directory
cd  <repo name>
```

- optional: create a secrets.env file with your WANDB API key, if you want to log results of local runs to W&B
  - the API key can be found/created when you login on www.wandb.ai, then go to www.wandb.ai/settings, and look under the header 'API keys'

```sh
echo "WANDB_API_KEY=_REPLACE_WITH_API_KEY_" > secrets.env
```


### Local

- to work with the code locally, run the following commands:

```sh
# use the following commands to install miniconda if you are on a Linux 64-bit machine, if you don't have it installed already
# for other systems, please follow instructions on https://docs.conda.io/en/latest/miniconda.html
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

# setup conda environment
conda env create -f conda.yaml

# activate environment
conda activate catchjoe

# IMPORTANT: install pre-commit
# make sure to run this, as it will prevent commits from being made if there are certain errors in the code
pre-commit install
```

## Usage

### Data

- to download the data, run the following command:

```sh
python -m src.data.make_dataset
```

### Features

- to extract features from the data, run the following command:

```sh
python -m src.features.build_features # If you want to overwrite the existing features use --overwrite
```

This will save the features in the `data/features` directory and pipeline object in the `outputs/models` directory.

### Models

- to train models and hyperparameter tune them, run the following command:

```sh
python -m src.models.train_model # if you want to use sweep hypertuning add --use_sweep
```

This will save the model in the `outputs/models` directory.

### Predictions

- to make predictions with a model, run the following command:

```sh
python -m src.models.predict_model  --input-file=./data/processed/test.json --output-file=./outputs/predictions/results.csv --model-file=../models/model_2xehd413.pkl
```

This will save the predictions in the `outputs/predictions` directory.


## Common issues

- below is a list of common issues and their solutions, if you encounter an issue and are able to resolve it, please add it here to help others who encounter the same issue
  - issue: `ModuleNotFoundError: No module named 'src'`
    - this is easily solved by running the following: `export PYTHONPATH=$PWD`
