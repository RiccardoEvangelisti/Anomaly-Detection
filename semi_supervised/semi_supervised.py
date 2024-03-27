from datetime import datetime
import logging, os
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

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
NAN_THRESH_PERCENT = 0.95

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
            target_names=["Normal data", "Anomalous data"],
        )
    )

    # # Build dataframe with predicted classes and original timestamps
    # pred_classes_test_ND = pd.DataFrame(pred_classes_test_ND, index=test_ND.index)
    # pred_classes_test_AD = pd.DataFrame(pred_classes_test_AD, index=test_AD.index)
    # pred_classes_test = (
    #     pd.concat((pred_classes_test_ND, pred_classes_test_AD), axis=0)
    #     .sort_index()
    #     .rename(columns={0: "nagiosdrained"})
    # )
    # pred_classes_test["timestamp"] = pd.to_datetime(df.loc[pred_classes_test.index]["timestamp"])

    # # Build dataframe with original classes (nagiosdrained) and original timestamps
    # classes_test = df.loc[np.concatenate((test_ND.index, test_AD.index))].sort_index()[["nagiosdrained", "timestamp"]]

    # # Plot
    # # This column is True if the prediction is correct and False otherwise
    # pred_classes_test["correct"] = pred_classes_test["nagiosdrained"] == classes_test["nagiosdrained"]
    # _, ax = plt.subplots(figsize=(16, 5))

    # sns.scatterplot(
    #     data=pred_classes_test,
    #     x="timestamp",
    #     y="nagiosdrained",
    #     hue="correct",
    #     palette=["red", "steelblue"],
    #     size="correct",
    #     sizes=(70, 100),
    #     style="correct",
    #     markers=["X", "."],
    #     edgecolor="none",
    #     ax=ax,
    # )

    # ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    # ax.tick_params(axis="x", labelrotation=10)
    # ax.set_ylabel("nagiosdrained")
    # ax.set_yticks([0, 1])
    # ax.legend(labels=["Correctly predicted", "Incorrectly predicted"])

    # plt.title("Test data: Correct vs Incorrect Predictions")
    # plt.tight_layout()
    # plt.show()

    # ################################################################

    # # Build dataframe with all original classes (nagiosdrained) and original timestamps
    # all_classes = df[["nagiosdrained", "timestamp"]].copy()

    # # Create a new column 'correct' in the dataframe 'all_classes'
    # # This column is 0 if the prediction is correct, 1 if incorrect and 2 for points that are not in the test set
    # all_classes.loc[pred_classes_test.index, "correct"] = np.where(
    #     all_classes.loc[pred_classes_test.index, "nagiosdrained"] == pred_classes_test["nagiosdrained"],
    #     0,
    #     1,
    # )
    # all_classes.loc[~all_classes.index.isin(pred_classes_test.index), "correct"] = 2

    # # Create a new subplot for the graph
    # _, ax = plt.subplots(figsize=(16, 5))

    # # Use seaborn to create the scatter plot
    # sns.scatterplot(
    #     data=all_classes,
    #     x="timestamp",
    #     y="nagiosdrained",
    #     hue="correct",
    #     palette={0: "steelblue", 1: "red", 2: "grey"},
    #     size="correct",
    #     sizes={0: 50, 1: 100, 2: 20},
    #     style="correct",
    #     markers={0: ".", 1: "X", 2: "."},
    #     edgecolor="none",
    #     ax=ax,
    # )

    # ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    # ax.tick_params(axis="x", labelrotation=10)
    # ax.set_ylabel("nagiosdrained")
    # ax.set_yticks([0, 1])
    # ax.legend(labels=['Train+Val data', 'Test data correctly predicted', 'Test data incorrectly predicted'])

    # # Set the title and show the plot
    # plt.title("All data: Correct vs Incorrect Test Predictions vs Not in test set")
    # plt.tight_layout()
    # plt.show()

    # Plot the results
    # Build a dataframe, for test ND+AD, with original indexes, predicted classes, original timestamps, boolean column to indentify correct/incorrect predictions
    pred_classes_test = (
        pd.concat(
            [
                pd.DataFrame(pred_classes_test_ND, index=test_ND.index),
                pd.DataFrame(pred_classes_test_AD, index=test_AD.index),
            ]
        )
        .sort_index()
        .rename(columns={0: "nagiosdrained"})
    )
    pred_classes_test["timestamp"] = pd.to_datetime(df.loc[pred_classes_test.index, "timestamp"])
    pred_classes_test["correct"] = (
        pred_classes_test["nagiosdrained"] == df.loc[pred_classes_test.index, "nagiosdrained"]
    )

    # Plot only test values
    _, ax = plt.subplots(2, 1, figsize=(16, 10))
    sns.scatterplot(
        data=pred_classes_test,
        x="timestamp",
        y="nagiosdrained",
        hue="correct",
        palette=["red", "steelblue"],
        size="correct",
        sizes=(70, 100),
        style="correct",
        markers=["X", "."],
        edgecolor="none",
        ax=ax[0],
    )
    ax[0].xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax[0].tick_params(axis="x", labelrotation=10)
    ax[0].set_ylabel("nagiosdrained")
    ax[0].set_yticks([0, 1])
    ax[0].legend(labels=["Test data Correctly predicted", "Test data Incorrectly predicted"])
    ax[0].set_title("Test data: Correct vs Incorrect Predictions")

    # Plot all data
    df["correct"] = 2  # default to '2' for Train+Val data
    df.loc[pred_classes_test.index, "correct"] = pred_classes_test["correct"].map({True: 0, False: 1})

    sns.scatterplot(
        data=df,
        x="timestamp",
        y="nagiosdrained",
        hue="correct",
        palette={0: "steelblue", 1: "red", 2: "grey"},
        size="correct",
        sizes={0: 50, 1: 100, 2: 20},
        style="correct",
        markers={0: ".", 1: "X", 2: "."},
        edgecolor="none",
        ax=ax[1],
    )
    ax[1].xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax[1].tick_params(axis="x", labelrotation=10)
    ax[1].set_ylabel("nagiosdrained")
    ax[1].set_yticks([0, 1])
    ax[1].legend(
        title="All points are displayed in their original position",
        title_fontsize="10",
        labels=["Train+Val data", "Test data correctly predicted", "Test data incorrectly predicted"],
    )
    plt.title("All original nagiosdrained points")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
