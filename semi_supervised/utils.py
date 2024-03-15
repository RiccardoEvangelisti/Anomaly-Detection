import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from keras import regularizers
from keras.layers import Dense, Input
from keras.models import Model

from query_tool.query_tool import M100DataClient


def build_dataset(node, dataset_path):
    if not isinstance(node, str):
        raise ValueError("node must be a string")
    
    os.path.isdir(dataset_path)
    client = M100DataClient(dataset_path)
    
    df = client.query_plugins(plugins="nagios", node=node).sort_values(by="timestamp", ascending=True)
    
    """- Build the complete dataset, with anomalies and with nagios column"""
    return df


def split_df(df, train=60, val=10, test=20, rand=None):
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
    autoencoder.fit(
        train_ND,
        train_ND,  # train_ND as both the input and the target since this is a reconstruction model
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(val_ND, val_ND),
        # sample_weight=np.asarray(active_idle), # not used for now
        verbose=1,
    )
    return autoencoder


def autoencoder_predict(autoencoder: Model, actual: pd.DataFrame, name):
    decoded = autoencoder.predict(actual)
    decoded = pd.DataFrame(decoded, index=actual.index)
    print("Reconstruction MSE on {}: {}".format(name, mean_squared_error(actual, decoded)))
    return decoded


def calculate_threshold(val_ND: np.ndarray, decoded_val_ND: np.ndarray, val_AD: np.ndarray, decoded_val_AD: np.ndarray):

    if val_AD is None or decoded_val_AD is None:
        raise ValueError("Invalid value for val_AD or decoded_val_AD")
    if val_ND is None or decoded_val_ND is None or not val_ND.any() or not decoded_val_ND.any():
        raise ValueError("Invalid value for val_AD or decoded_val_AD")

    # A list in which each element corresponds to the maximum absolute error of one row. **It's the maximum error that an observation has made**
    max_errors_list_valid_ND = np.max(np.abs(decoded_val_ND - val_ND), axis=1)
    max_errors_list_valid_AD = np.max(np.abs(decoded_val_AD - val_AD), axis=1)

    classes = np.concatenate(([0] * val_ND.shape[0], [1] * val_AD.shape[0]))
    errors = np.concatenate((max_errors_list_valid_ND, max_errors_list_valid_AD))

    n_perc_min = 70
    n_perc_max = 100
    best_n_perc = n_perc_max
    fscore_val_best = 0
    n_percs = []
    precs = []
    recalls = []
    fscores = []

    for n_perc in range(n_perc_min, n_perc_max + 1):
        error_threshold = np.percentile(max_errors_list_valid_ND, n_perc)

        predictions = []
        for e in errors:
            if e > error_threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(
            classes, predictions, average="weighted"
        )

        fscores.append(fscore_val)
        precs.append(precision_val)
        recalls.append(recall_val)
        n_percs.append(n_perc)

        if fscore_val > fscore_val_best:
            fscore_val_best = fscore_val
            best_n_perc = n_perc

    best_error_threshold = np.percentile(max_errors_list_valid_ND, best_n_perc)
    print("Best threshold on validation data: {}".format(best_error_threshold))

    mrkrsize = 5
    fig = plt.figure()
    plt.plot(n_percs, fscores, c="b", label="F-Score", marker="o", linewidth=2, markersize=mrkrsize)
    plt.plot(n_percs, recalls, c="g", label="Recall", marker="x", linewidth=2, markersize=mrkrsize)
    plt.plot(n_percs, precs, c="r", label="Precision", marker="D", linewidth=2, markersize=mrkrsize)
    plt.axvline(x=best_n_perc, c="grey", linestyle="--")
    plt.xlabel("N-th percentile")
    plt.ylabel("Detection Accuracy")
    plt.legend()
    plt.savefig("./semi_supervised/outputs/threshold_validation")
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

    precision, recall, fscore, _ = precision_recall_fscore_support(classes, predictions, average="weighted")
    
    return classes, precision, recall, fscore