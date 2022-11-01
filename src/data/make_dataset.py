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
    # read googledrive id from vars.env
    gdrive_id = os.environ.get("DATA_GDRIVE_ID")
    output_path = paths.RAW_DATASETS_DIR
    # check if data is already downloaded
    if (output_path / "dataset.json").exists() and (output_path / "verify.json").exists():
        logger.info("Data already downloaded")
        return

    # download data
    try:
        subprocess.run(
            [
                "gdown",
                f"https://drive.google.com/uc?id={gdrive_id}",
                "-O",
                f"{str(output_path.resolve())}/catchjoe.zip",
            ]
        )
        subprocess.run(
            [
                "unzip",
                f"{str(output_path.resolve())}/catchjoe.zip",
                "-d",
                f"{str(output_path.resolve())}/",
            ]
        )
        subprocess.run(
            [
                "rm",
                "-rf",
                f"{str(output_path.resolve())}/catchjoe.zip",
            ]
        )

        logger.info(f"Data downloaded to {output_path}")
    except Exception as e:
        logger.error(f"Could not download data from Google Drive: {e}")
        raise e


# preprocess data
def combine_multiple_sites(sites):
    temp = {}
    for s in sites:
        if s["site"] not in temp:
            temp[s["site"]] = s["length"]
        else:
            temp[s["site"]] += s["length"]
    return temp


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the data to be used in the model.

    Args:
        df (pd.DataFrame): Dataframe to be transformed.

    Returns:
        pd.DataFrame: Transformed dataframe.
    """
    # aggregate the sites by summing the length of the sites for each row
    df.sites = df.sites.apply(combine_multiple_sites, 1)
    return df


def preproces_data(overwrite=False):
    """Preprocess data.
    It uses the data from the raw folder and saves the preprocessed data to the
    processed folder.
    In the future, this function can be replaced by a pipeline.
    """

    if (paths.PROCESSED_DATASETS_DIR / "dataset.json").exists() and (paths.PROCESSED_DATASETS_DIR / "verify.json").exists():
        if overwrite:
            logger.info("Processed data already exists. Overwriting...")
        else:
            logger.info("Processed data already exists. Skipping...")
            return

    logger.info("Preprocessing data")
    # read data
    data = pd.read_json(paths.RAW_DATASETS_DIR / "dataset.json")
    verify = pd.read_json(paths.RAW_DATASETS_DIR / "verify.json")

    # we aggregate the sites by summing the length of the sites for each row
    data = transform_data(data)
    verify = transform_data(verify)

    # save data
    data.to_json(paths.PROCESSED_DATASETS_DIR / "dataset.json")
    verify.to_json(paths.PROCESSED_DATASETS_DIR / "verify.json")
    logger.info(f"Data preprocessed and saved to {paths.PROCESSED_DATASETS_DIR}")


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
