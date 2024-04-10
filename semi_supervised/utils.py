from datetime import timedelta
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from keras import regularizers
from keras.layers import Dense, Input
from keras.models import Model


def build_dataset(plugins, node, dataset_rebuild_path, NAN_THRESH_PERCENT):
    # Merge all the different plugin datasets
    df = pd.DataFrame()
    for plugin in plugins:
        df_temp = pd.read_csv(dataset_rebuild_path + plugin + "_rebuild" + "_node" + node + ".csv")
        if df.empty:
            df = df_temp
        else:
            df = pd.merge(df, df_temp, on=["timestamp"], how="outer")

    df = df.sort_values(by="timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract useful informations
    original_num_rows = df.shape[0]
    original_columns = df.columns.to_list()
    original_num_rows_per_column = df.count()

    # Removing rows with missing nagios value
    df = df[df["nagiosdrained"].notna()]

    # Keeping features that have present at least NAN_THRESH_PERCENT values of the total dataframe size (i.e. the 80% of non-nan values)
    df = df.dropna(thresh=df.shape[0] * NAN_THRESH_PERCENT, axis=1)

    # Drop the remaining NaN rows
    df = df.dropna()

    # Print description
    print("\n-----------------------------------------------------------")
    print(
        "Number of rows:\toriginal({}) − removed({}) = {} rows".format(
            original_num_rows, original_num_rows - df.shape[0], df.shape[0]
        )
    )
    print(
        "Number of cols:\toriginal({}) − removed({}) = {} cols".format(
            len(original_columns), len(original_columns) - df.shape[1], df.shape[1]
        )
    )
    print("\nRemoved\t| Original rows\t|  Columns")
    print("-------\t| -------------\t|  ------")
    for col in original_columns:
        print("  {}\t|\t{}\t|  {}".format("no" if col in df.columns else "YES", original_num_rows_per_column[col], col))

    return df.reset_index(drop=True)


def extract_anomalous_data(df: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """Separate the anomaly periods, based on the "nagiosdrained" flag: if equals to 1, it's anomaly period."""
    df_ND_indexes = df.loc[df["nagiosdrained"] == 0].index
    df_AD_indexes = df.loc[df["nagiosdrained"] == 1].index
    return df_ND_indexes, df_AD_indexes


def split_df(df, train, val, test, rand=None):
    if train + val + test != 100:
        raise ValueError("The sum must be 100")
    temp, test_data = train_test_split(df, test_size=test / 100.0, random_state=rand)
    train_data, val_data = (
        (None, temp) if train == 0 else train_test_split(temp, test_size=val / (100 - test), random_state=rand)
    )
    return train_data, val_data, test_data


def move_almost_AD(df_ND: pd.DataFrame, df_AD: pd.DataFrame, delta, train_ND, val_ND, test_ND):
    for index, normal_sample in df_ND.iterrows():
        if index not in test_ND.index:
            if df_AD.loc[df_AD["timestamp"] == normal_sample.timestamp + delta].shape[0] > 0:
                if index in train_ND.index:
                    test_ND = pd.concat((test_ND, train_ND.loc[[index]]), axis=0)
                    train_ND = train_ND.drop(index)
                if index in val_ND.index:
                    test_ND = pd.concat((test_ND, val_ND.loc[[index]]), axis=0)
                    val_ND = val_ND.drop(index)

    return train_ND, val_ND, test_ND


def model_definition(n_features: int, train_ND: pd.DataFrame, val_ND: pd.DataFrame, EPOCHS: int, BATCH_SIZE: int):
    input_data = Input(shape=(n_features,))
    encoded = Dense(n_features * 10, activation="relu", activity_regularizer=regularizers.l1(1e-5))(input_data)
    decoded = Dense(n_features, activation="linear")(encoded)

    autoencoder = Model(input_data, decoded)
    autoencoder.compile(optimizer="adam", loss="mean_absolute_error")
    history = autoencoder.fit(
        train_ND,
        train_ND,  # train_ND as both the input and the target since this is a reconstruction model
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(val_ND, val_ND),
        verbose=0,
    )
    return history, autoencoder


def autoencoder_predict(autoencoder: Model, actual: pd.DataFrame, name):
    decoded = autoencoder.predict(actual, verbose=0)
    decoded = pd.DataFrame(decoded, columns=actual.columns, index=actual.index)
    print("Reconstruction MSE on {}:\t{}".format(name, mean_squared_error(actual, decoded)))
    return decoded


def calculate_threshold(val_ND: np.ndarray, decoded_val_ND: np.ndarray, val_AD: np.ndarray, decoded_val_AD: np.ndarray):
    if val_AD is None or decoded_val_AD is None:
        raise ValueError("Invalid value for val_AD or decoded_val_AD")
    if val_ND is None or decoded_val_ND is None or val_ND.size <= 0 or decoded_val_ND.size <= 0:
        raise ValueError("Invalid value for val_AD or decoded_val_AD")

    # A list in which each element corresponds to the maximum absolute error of one row. **It's the maximum error that each observation has made**
    max_errors_list_valid_ND = np.max(np.abs(decoded_val_ND - val_ND), axis=1)
    max_errors_list_valid_AD = np.max(np.abs(decoded_val_AD - val_AD), axis=1)

    classes = np.concatenate(([0] * val_ND.shape[0], [1] * val_AD.shape[0]))
    errors = np.concatenate((max_errors_list_valid_ND, max_errors_list_valid_AD))

    n_perc_min = 1
    n_perc_max = 100
    best_n_perc = n_perc_max
    fscore_val_best = 0
    n_percs = []
    precs = []
    recalls = []
    fscores = []

    # for each percentage
    for n_perc in range(n_perc_min, n_perc_max + 1):
        # Returns the q-th percentile of "max_errors_list_valid_ND", as the value q/100 of the way from the minimum to the maximum of a *sorted* copy of the list. This function is the same as the median if q=50, the same as the minimum if q=0 and the same as the maximum if q=100.
        error_threshold = np.percentile(max_errors_list_valid_ND, n_perc)

        predictions = []
        for e in errors:
            if e > error_threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        if sum(map(lambda x: x == 1, predictions)) == 0:
            print("WARNING: no predictions of class 1 with percentile:", n_perc)

        precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(
            classes, predictions, average="weighted", zero_division=0
        )

        fscores.append(fscore_val)
        precs.append(precision_val)
        recalls.append(recall_val)
        n_percs.append(n_perc)

        if fscore_val > fscore_val_best:
            fscore_val_best = fscore_val
            best_n_perc = n_perc

    best_error_threshold = np.percentile(max_errors_list_valid_ND, best_n_perc)
    print("\nBest threshold on validation data: {}%".format(best_n_perc))

    mrkrsize = 5
    plt.title("Scores on validation data")
    plt.plot(n_percs, fscores, c="b", label="F-Score", marker="o", markersize=mrkrsize)
    plt.plot(n_percs, recalls, c="g", label="Recall", marker="x", markersize=mrkrsize)
    plt.plot(n_percs, precs, c="r", label="Precision", marker="D", markersize=mrkrsize)
    plt.axvline(x=best_n_perc, c="grey", linestyle="--")
    plt.xlabel("N-th percentile")
    plt.ylabel("Score value")
    plt.legend()
    plt.show()

    return best_error_threshold, n_perc


def evaluate_model(normal_data: bool, test: np.ndarray, decoded_test: np.ndarray, threshold: float):
    """
    UNUSED
    """
    classes = [0 if normal_data else 1] * test.shape[0]
    errors = np.max(np.abs(decoded_test - test), axis=1)

    predictions = []
    for e in errors:
        if e > threshold:
            predictions.append(1)
        else:
            predictions.append(0)

    precision, recall, fscore, _ = precision_recall_fscore_support(
        classes, predictions, average="binary", pos_label=0 if normal_data else 1  # , zero_division=0
    )

    return predictions, precision, recall, fscore


def classify_data(normal_data: bool, test: np.ndarray, decoded_test: np.ndarray, threshold: float):
    classes = [0 if normal_data else 1] * test.shape[0]
    errors = np.max(np.abs(decoded_test - test), axis=1)

    predictions = []
    for e in errors:
        if e > threshold:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions, classes


def detect_AD_false_positives(pred_classes_test_ND: pd.DataFrame, df_AD: pd.DataFrame, delta: timedelta):
    # Extract only the incorrectly predicted normal data
    false_positives = pred_classes_test_ND.loc[pred_classes_test_ND["is_correct"] == False].sort_index()

    for fp in false_positives.itertuples():
        # Select the anomaly observations that are between the false positive and the false positive + delta, if any
        anomaly_after_fp = df_AD[(df_AD["timestamp"] > fp.timestamp) & (df_AD["timestamp"] <= fp.timestamp + delta)]
        # If there are any, the false positive has preceded by delta-time the real anomaly
        if not anomaly_after_fp.empty:
            print(
                f"ANOMALY PREDICTED AT: {fp.timestamp}, FIRST REAL ANOMALY AT: {anomaly_after_fp.iloc[0].timestamp} ({anomaly_after_fp.iloc[0].timestamp - fp.timestamp} before)"
            )


def build_pred_class_test(pred_classes_test_ND, pred_classes_test_AD, test_ND, test_AD, df):
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
    pred_classes_test["is_correct"] = (
        pred_classes_test["nagiosdrained"] == df.loc[pred_classes_test.index, "nagiosdrained"]
    )
    pred_classes_test["timestamp"] = df.loc[pred_classes_test.index, "timestamp"]
    return pred_classes_test