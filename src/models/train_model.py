#!/usr/bin/env python
# This scripts trains a model using the data in the features folder
# and saves the model in the models folder

import pickle
import sys
from pathlib import Path

import numpy as np
import typer
import wandb
from flaml import AutoML

from src.enums import DatasetType, RunType
from src.logger import root_logger
from src.paths import paths


sys.path.append("../")
import json
import time

import sklearn.metrics as metrics
from dotenv import load_dotenv
from scipy import sparse
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.experiment_tracking import init_wandb_run, init_wandb_sweep


load_dotenv("secrets.env")
# and the WANDB_PROJECT and WANDB_ENTITY from the vars.env file
load_dotenv("vars.env")


logger = root_logger.getChild(__name__)

target_metric = "f1"


def classification_metrics(y_true, y_pred):
    """
    Computes weighted precision, recall, f1-score for each class.
    """
    # Compute precision, recall, f1-score for each class
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average="macro")
    # Compute accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


def custom_metric(
    X_val,  # noqa: N803
    y_val,
    estimator,
    labels,
    X_train,  # noqa: N803
    y_train,
    weight_test=None,
    weight_train=None,
    config=None,
    groups_val=None,
    groups_train=None,
):
    """
    Custom metric for AutoML

    This function is used to compute the metric for AutoML. It is needed to loging cross validation scores to wandb.
    """

    global target_metric
    start = time.time()
    y_pred = estimator.predict(X_val)
    pred_time = (time.time() - start) / X_val.shape[0]
    val_precision, val_recall, val_f1, val_accuracy = classification_metrics(y_val, y_pred)

    y_pred = estimator.predict(X_train)
    train_precision, train_recall, train_f1, train_accuracy = classification_metrics(y_train, y_pred)

    # check the loss function
    if target_metric == "f1":
        val_loss = 1 - val_f1
        train_loss = 1 - train_f1
    elif target_metric == "accuracy":
        val_loss = 1 - val_accuracy
        train_loss = 1 - train_accuracy
    elif target_metric == "precision":
        val_loss = 1 - val_precision
        train_loss = 1 - train_precision
    elif target_metric == "recall":
        val_loss = 1 - val_recall
        train_loss = 1 - train_recall
    elif target_metric == "log_loss":
        val_loss = metrics.log_loss(y_val, y_pred)
        train_loss = metrics.log_loss(y_train, y_pred)
    else:
        raise ValueError("unknown target metric")

    calculated_metrics = {
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_accuracy": val_accuracy,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "train_accuracy": train_accuracy,
        "pred_time": pred_time,
        "train_loss": train_loss,
    }
    wandb.log(calculated_metrics)
    return val_loss, calculated_metrics


def get_X_y(metadata_path: Path, dataset_type: DatasetType = DatasetType.TRAIN) -> tuple:  # noqa: N802
    """
    Get X and y from metadata
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    X = sparse.load_npz(metadata["features_path"])
    y = None
    if dataset_type == DatasetType.TRAIN:
        y = np.array(metadata["target"]).reshape(-1, 1)

    return X, y


def fit_model(X, y, tm, automl_settings):  # noqa: N803
    """
    Fit model to data
    """

    global target_metric
    target_metric = tm

    automl = AutoML()

    automl.fit(X, y, **automl_settings)
    return automl


def train_model(use_sweep: bool = True):
    """
    Train model and save it to model_path
    """

    config_defaults = {
        "estimator": "lgbm",
        "target_metric": "f1",
    }
    wandb_run_id = init_wandb_run(model_name="automl", run_type=RunType.TRAINING, run_tag="trial_01", config=config_defaults)

    metadata_path = paths.FEATURES_DATASETS_DIR / "metadata.json"
    X_train, y_train = get_X_y(metadata_path)

    logger.info(f"Size of X_train: {X_train.shape}, y_train: {y_train.shape}")
    # # validation set
    # metadata_path = paths.FEATURES_DATASETS_DIR / "valid_metadata.json"
    # X_val, _ = get_X_y(metadata_path)

    if use_sweep:
        estimator_list = [wandb.config["estimator"]]
        target_metric_selected = wandb.config["target_metric"]

        logger.info(f"Training model with config: {wandb.config}")

    else:
        estimator_list = ["lgbm", "xgboost"]
        target_metric_selected = "f1"
        logger.info(f"Training with default parameters w/o wandb hypertuning")
    # train model
    automl_settings = {
        "time_budget": 30,
        "metric": custom_metric,
        "task": "classification",
        "estimator_list": estimator_list,
        "verbose": 3,
        "retrain_full": True,
        "log_training_metric": True,
        "keep_search_state": True,
        "auto_augment": True,
        "split_type": "stratified",
        "early_stop": True,
        "n_splits": 10,
    }

    automl = fit_model(X_train, y_train, target_metric_selected, automl_settings)

    wandb.log({"val_loss": automl.best_loss})

    # get predictioss and metrics
    y_pred = automl.predict(X_train)

    # wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_train.tolist(), preds=y_pred.tolist(), class_names=[0, 1])}) # thera was a type error here TODO: fix later

    precision, recall, f1, accuracy = classification_metrics(y_train, y_pred)
    logger.info(f"Train metrics: precision: {precision}, recall: {recall}, f1: {f1}, accuracy: {accuracy}")
    print(confusion_matrix(y_train, y_pred))
    print(classification_report(y_train, y_pred))

    wandb.log(
        {
            "train_precision": precision,
            "train_recall": recall,
            "train_f1": f1,
            "train_accuracy": accuracy,
        }
    )

    # save model
    model_path = paths.MODELS_DIR / f"model_{wandb_run_id}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(automl, f)


def main(use_sweep: bool = False):
    if use_sweep:
        wandb_sweep_id = init_wandb_sweep()

        wandb.agent(wandb_sweep_id, function=train_model, count=5)
    else:
        train_model(use_sweep=False)


if __name__ == "__main__":
    # force to relogin to wandb
    typer.run(main)

    # get the best model
