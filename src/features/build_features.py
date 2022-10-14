# This script is to define the methods and functions to build features
#
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer
from dotenv import find_dotenv, load_dotenv

from src.enums import DatasetType
from src.logger import root_logger
from src.paths import paths
from src.utils.decorators import log_timing


logger = root_logger.getChild(__name__)


# methods to build features where return the features, the target if any, and the name of the features as dict
@log_timing
def build_features(df: pd.DataFrame, overwrite: bool = False, dataset_type: DatasetType = DatasetType.TRAIN) -> dict:
    """Build features from the raw data.
    Args:
        df (pd.DataFrame): Raw data
    Returns:
        dict: Features, target, and feature names
    """
    features = np.array([])
    feature_names: List[str] = []

    return {"features": features, "feature_names": list(feature_names)}


def main(overwrite: bool = False):
    logger.info(f"Building features")

    features_path = paths.FEATURES_DATASETS_DIR / "features.npz"
    metadata_path = paths.FEATURES_DATASETS_DIR / "metadata.json"

    logger.info(f"Done building features")


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv("vars.env"))

    # # run with typer
    typer.run(main)
