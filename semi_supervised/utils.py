from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from keras import regularizers
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler


def build_dataset(plugins, node, dataset_rebuild_path, NAN_THRESH_PERCENT):
    # Aggregating the dataset
    df = pd.DataFrame()
    for plugin in plugins:
        df_temp = pd.read_csv(dataset_rebuild_path + plugin + "_rebuild" + "_node:" + node + ".csv")
        if df.empty:
            df = df_temp
        else:
            df = pd.merge(df, df_temp, on=["timestamp"], how="outer")

    df = df.sort_values(by="timestamp").reset_index(drop=True)

    # Removing rows with missing nagios value
    df = df[df["nagiosdrained"].notna()]

    # Keeping features that have present at least NAN_THRESH_PERCENT values of the total dataframe size (i.e. the 80% of non-nan values)
    df = df.dropna(thresh=df.shape[0] * NAN_THRESH_PERCENT, axis=1)

    # Drop the remaining NaN rows
    df = df.dropna()

    return df


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
        # sample_weight=np.asarray(active_idle), # not used for now
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
    n_perc_max = 99
    best_n_perc = n_perc_max
    fscore_val_best = 0
    n_percs = []
    precs = []
    recalls = []
    fscores = []

    # for each percentage
    for n_perc in range(n_perc_min, n_perc_max + 1):
        # Returns the q-th percentile of "max_errors_list_valid_ND", as the value q/100 of the way from the minimum to the maximum of a *sorte* copy of the list. This function is the same as the median if q=50, the same as the minimum if q=0 and the same as the maximum if q=100.
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
    print("\nBest threshold on validation data: {}".format(best_error_threshold))

    mrkrsize = 5
    plt.title("Scores on validation data")
    plt.plot(n_percs, fscores, c="b", label="F-Score", marker="o", linewidth=2, markersize=mrkrsize)
    plt.plot(n_percs, recalls, c="g", label="Recall", marker="x", linewidth=2, markersize=mrkrsize)
    plt.plot(n_percs, precs, c="r", label="Precision", marker="D", linewidth=2, markersize=mrkrsize)
    plt.axvline(x=best_n_perc, c="grey", linestyle="--")
    plt.xlabel("N-th percentile")
    plt.ylabel("Score value")
    plt.legend()
    plt.show()

    return best_error_threshold, n_perc


def evaluate_model(normal_data: bool, test: np.ndarray, decoded_test: np.ndarray, threshold: float):
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
