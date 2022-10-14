#!/usr/bin/env python
# when provided a json file convert it to a dataframe, then transform it and predict the result
# json file should be in the format of the test data

import pickle
from pathlib import Path

import pandas as pd
import typer

from src.data.make_dataset import transform_data
from src.enums import DatasetType
from src.features.build_features import build_features
from src.logger import root_logger
from src.models.train_model import custom_metric  # noqa: F401 # IGNORE THIS LINE
from src.paths import paths
from src.utils.utils import id2class


logger = root_logger.getChild(__name__)


def process_data(input_file: Path):
    # load json file
    df = pd.read_json(input_file)
    df_return = df.copy()
    # preprocess data
    df = transform_data(df)
    # build features
    data_dict = build_features(df, overwrite=False, dataset_type=DatasetType.TEST)
    X = data_dict["features"]
    return df_return, X


def load_model(path: Path = paths.BEST_MODEL_PATH):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def predict(
    input_file: Path = paths.RAW_DATASETS_DIR / "test.json", output_file: Path = paths.OUTPUTS_DIR / "results.csv", model_file: Path = paths.BEST_MODEL_PATH
):
    # load model and check if it exists
    if not model_file.exists():
        logger.error(f"Model file {model_file} does not exist")
        raise FileNotFoundError
    model = load_model(model_file)
    # process data
    if not input_file.exists():
        logger.error(f"Input file {input_file} does not exist")
        raise FileNotFoundError

    df, X = process_data(input_file)

    # predict
    predictions = model.predict(X)
    predictions = [id2class[pred] for pred in predictions]

    # save predictions
    df["prediction"] = predictions
    df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")


if __name__ == "__main__":

    # Typer allows the script to be called with command line arguments
    # if a parameter has no default value, it must be passed as an argument
    # if a parameter has a default value, it may be passed as a flag (e.g. --parameter)
    # Example usage:
    # python -m src.models.predict_model  --input-file=./data/raw/verify.json --output-file=./outputs/predictions/results.csv --model-file=./outputs/models/model_best.pkl

    typer.run(predict)
