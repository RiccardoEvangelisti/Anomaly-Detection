from datetime import datetime
import logging, os
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler

logging.disable(logging.WARNING)  # disable TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
import matplotlib.pyplot as plt
import pandas as pd

from utils import (
    autoencoder_predict,
    build_dataset,
    calculate_threshold,
    classify_data,
    evaluate_model,
    extract_anomalous_data,
    model_definition,
    split_df,
)

"""
ND: Normal Data
AD: Anomalous Data
"""

YEAR = 2022
MONTH = 9
date_dataset = datetime(YEAR, MONTH, 1)

DATASET_FOLDER = "./dataset/"
DATASET_FOLDER_REBUILD = DATASET_FOLDER + "rebuild/"
dataset_rebuild_path = DATASET_FOLDER_REBUILD + date_dataset.strftime("%y-%m") + "/"

NODE = "10"

ACCEPTED_PLUGINS = ["nagios", "ganglia", "ipmi"]
NAN_THRESH_PERCENT = 0.8

RANDOM_STATE = 42
TRAIN_ND_PERC, VAL_ND_PERC, TEST_ND_PERC = 60, 10, 30
VAL_AD_PERC, TEST_AD_PERC = 10, 90  # 30, 70

EPOCHS = 256
BATCH_SIZE = 64


def main():
    keras.utils.set_random_seed(RANDOM_STATE)

    # Build dataset
    df = build_dataset(ACCEPTED_PLUGINS, NODE, dataset_rebuild_path, NAN_THRESH_PERCENT)

    # Extract anomalous data
    df_ND_indexes, df_AD_indexes = extract_anomalous_data(df)

    # Split ND data, without "timestamp" and "nagiosdrained" features
    train_ND, val_ND, test_ND = split_df(
        df.loc[df_ND_indexes].drop(columns=["timestamp", "nagiosdrained"]),
        train=TRAIN_ND_PERC,
        val=VAL_ND_PERC,
        test=TEST_ND_PERC,
        rand=RANDOM_STATE,
    )

    # Split AD actual data, without "timestamp" and "nagiosdrained" features
    _, val_AD, test_AD = split_df(
        df.loc[df_AD_indexes].drop(columns=["timestamp", "nagiosdrained"]),
        train=0,
        val=VAL_AD_PERC,
        test=TEST_AD_PERC,
        rand=RANDOM_STATE,
    )

    # Fit the Scaler only on ND train data
    scaler = MinMaxScaler().fit(train_ND)
    # Scale all the data
    train_ND = pd.DataFrame(scaler.transform(train_ND), columns=train_ND.columns, index=train_ND.index)
    val_ND = pd.DataFrame(scaler.transform(val_ND), columns=val_ND.columns, index=val_ND.index)
    test_ND = pd.DataFrame(scaler.transform(test_ND), columns=test_ND.columns, index=test_ND.index)
    val_AD = pd.DataFrame(scaler.transform(val_AD), columns=val_AD.columns, index=val_AD.index)
    test_AD = pd.DataFrame(scaler.transform(test_AD), columns=test_AD.columns, index=test_AD.index)

    # Autoencoder definition
    n_features = df.shape[1] - 2  # minus the "timestamp" and "nagiosdrained" features
    history, autoencoder = model_definition(
        n_features,
        np.asarray(train_ND).astype(np.float64),  # conversion to array
        np.asarray(val_ND).astype(np.float64),  # conversion to array
        EPOCHS,
        BATCH_SIZE,
    )

    plt.title("Autoencoder fitting")
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Error")
    plt.legend()
    plt.show()
    print("\n-----------------------------------------------------------")

    # Prediction of normal data
    _ = autoencoder_predict(autoencoder, train_ND, "ND train")
    decoded_test_ND = autoencoder_predict(autoencoder, test_ND, "ND test")
    decoded_val_ND = autoencoder_predict(autoencoder, val_ND, "ND val")

    # Prediction of anomalous data
    decoded_test_AD = autoencoder_predict(autoencoder, test_AD, "AD test")
    decoded_val_AD = autoencoder_predict(autoencoder, val_AD, "AD val")

    # Find best Threshold using validation sets
    threshold, _ = calculate_threshold(
        val_ND,
        decoded_val_ND,
        val_AD,
        decoded_val_AD,
    )

    print("\n-----------------------------------------------------------")
    
    # Classify unseen ND data
    pred_classes_test_ND, actual_classes_test_ND = classify_data(True, test_ND, decoded_test_ND, threshold)
    # Classify unseen AD data
    pred_classes_test_AD, actual_classes_test_AD = classify_data(False, test_AD, decoded_test_AD, threshold)

    print(
        classification_report(
            actual_classes_test_ND + actual_classes_test_AD,
            pred_classes_test_ND + pred_classes_test_AD,
            output_dict=False,
            digits=4,
            target_names=["0: Normal data", "1: Anomalous data"],
        )
    )

    # Build dataframe with predicted classes and original timestamps
    pred_classes_test_ND = pd.DataFrame(pred_classes_test_ND, index=test_ND.index)
    pred_classes_test_AD = pd.DataFrame(pred_classes_test_AD, index=test_AD.index)
    pred_classes_test = (
        pd.concat((pred_classes_test_ND, pred_classes_test_AD), axis=0)
        .sort_index()
        .rename(columns={0: "nagiosdrained"})
    )
    pred_classes_test["timestamp"] = df.loc[pred_classes_test.index]["timestamp"]

    # Build dataframe with original classes (nagiosdrained) and original timestamps
    classes_test = df.loc[np.concatenate((test_ND.index, test_AD.index))].sort_index()[["nagiosdrained", "timestamp"]]

    # Plot
    _, ax = plt.subplots(2, 1, figsize=(16, 10))
    for axes, array, color, label in zip(
        ax,
        [classes_test, pred_classes_test],
        ["blue", "orange"],
        ["Actual classes (nagiosdrained)", "Predicted classes"],
    ):
        axes.plot(array["timestamp"], array["nagiosdrained"], label=label, color=color)
        axes.set_xticks(array["timestamp"][::100])
        axes.tick_params(axis="x", labelrotation=30)
        axes.set_ylabel("nagiosdrained")
        axes.set_yticks([0, 1])
        axes.legend()

    plt.suptitle("Test data")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
