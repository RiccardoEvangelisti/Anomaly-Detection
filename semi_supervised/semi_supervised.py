import logging, os

logging.disable(logging.WARNING)  # disable TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datetime import datetime, timedelta
import calendar
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import keras

from utils import (
    autoencoder_predict,
    build_dataset,
    build_pred_class_test,
    calculate_threshold,
    classify_data,
    detect_AD_false_positives,
    extract_anomalous_data,
    model_definition,
    move_almost_AD,
    split_df,
)

"""
Abbreviations:
    - "ND": Normal Data
    - "AD": Anomalous Data
"""

YEAR = 2022
MONTH = 9

DATASET_FOLDER = "./dataset/"
DATASET_FOLDER_REBUILD = DATASET_FOLDER + "rebuild/"

NODE = "578"  # must be string

ACCEPTED_PLUGINS = ["nagios", "ganglia", "ipmi", "jobtable"]  # plugins to consider

# the percentage of non-NaN values over the entire length of the dataset that a column must have, or be dropped
NAN_THRESH_PERCENT = 0.8

RANDOM_STATE = 42

TRAIN_ND_PERC, VAL_ND_PERC, TEST_ND_PERC = 60, 10, 30  # split percentages of ND data
VAL_AD_PERC, TEST_AD_PERC = 10, 90  # 30, 70 # split percentages of AD data

DELTA_TIME_BEFORE_ANOMALY = timedelta(hours=2)  # delta-time before an anomaly to detect false positives

EPOCHS = 256  # autoencoder epochs
BATCH_SIZE = 64  # autoencoder batch size


def main():

    # Preliminary operations
    keras.utils.set_random_seed(RANDOM_STATE)
    dataset_rebuild_path = DATASET_FOLDER_REBUILD + datetime(YEAR, MONTH, 1).strftime("%y-%m") + "/"
    first_day = datetime(YEAR, MONTH, 1)
    last_day = calendar.monthrange(first_day.year, first_day.month)[1]
    all_days = np.arange(first_day, last_day, dtype="datetime64[D]")

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

    # Move to the test (ND) the "almost anomalous data", that is the ND observations "delta" time before an anomaly has been detected and flagged by nagios
    train_ND, val_ND, test_ND = move_almost_AD(
        df.iloc[df_ND_indexes], df.iloc[df_AD_indexes], DELTA_TIME_BEFORE_ANOMALY, train_ND, val_ND, test_ND
    )

    # Split AD actual data, without "timestamp" and "nagiosdrained" features
    # If the AD data are too few, the split can throw an error because the array "val_AD" would be empty -> increment the "VAL_AD_PERC"
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

    # Find the best Threshold using validation sets
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
    print("\n-----------------------------------------------------------")

    # Build a support dataframe of which:
    # the data is the concatenation of predicted classes of test ND and test AD
    # the index is the original indexes
    # the columns are: predicted classes, boolean column to identify correct/incorrect predictions, original timestamps
    pred_classes_test = build_pred_class_test(pred_classes_test_ND, pred_classes_test_AD, test_ND, test_AD, df)

    # Detect the false positive of the anomalous points, that are the false negatives of normal points
    detect_AD_false_positives(pred_classes_test.loc[test_ND.index], df.iloc[df_AD_indexes], DELTA_TIME_BEFORE_ANOMALY)

    # Plot all data
    df["is_correct"] = 2  # default to 2 for Train+Val data
    df.loc[pred_classes_test.index, "is_correct"] = pred_classes_test["is_correct"].map({True: 0, False: 1})

    _, ax = plt.subplots(figsize=(16, 5))
    sns.scatterplot(
        data=df,
        x="timestamp",
        y="nagiosdrained",
        hue="is_correct",
        palette={0: "steelblue", 1: "red", 2: "grey"},
        size="is_correct",
        sizes={0: 50, 1: 100, 2: 20},
        style="is_correct",
        markers={0: ".", 1: "X", 2: "."},
        edgecolor="none",
        ax=ax,
    )
    ax.set_xticks(all_days)
    ax.tick_params(axis="x", labelrotation=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_ylabel("nagiosdrained")
    ax.set_yticks([0, 1])
    ax.legend(
        title="All points are displayed in their original position",
        title_fontsize="10",
        labels=["Train+Val data", "Test data correctly predicted", "Test data incorrectly predicted"],
    )
    plt.title("All original nagiosdrained points")
    plt.tight_layout()
    plt.show()

    # Clean up after plotting
    df.drop(columns="is_correct")


if __name__ == "__main__":
    main()
