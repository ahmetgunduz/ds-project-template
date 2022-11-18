import os
import subprocess
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.logger import root_logger
from src.paths import paths


# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv("vars.env"))

logger = root_logger.getChild(__name__)


def download_data():
    """Download data from external source."""
    logger.info("Downloading data from external source")
    try:
        ## DOWNLOAD CODE HERE
        logger.info(f"Data downloaded")
    except Exception as e:
        logger.error(f"Could not download data from Google Drive: {e}")
        raise e


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the data to be used in the model.

    Args:
        df (pd.DataFrame): Dataframe to be transformed.

    Returns:
        pd.DataFrame: Transformed dataframe.
    """
    raise NotImplementedError


def preproces_data(df: pd.DataFrame, overwrite=False) -> None:
    """Preprocess data.
    It uses the data from the raw folder and saves the preprocessed data to the
    processed folder.
    In the future, this function can be replaced by a pipeline.
    """
    transform_data(df)
    raise NotImplementedError


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    # download the data
    download_data()

    # preprocess the data
    preproces_data(overwrite=True)


if __name__ == "__main__":

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
