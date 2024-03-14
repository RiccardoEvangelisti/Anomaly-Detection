import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import pytest

from task.semi_supervised.utils import (
    autoencoder_predict,
    calculate_threshold,
    split_df,
    model_definition,
    evaluate_model,
)


@pytest.fixture
def df_model():
    timestamps = pd.date_range(start=pd.Timestamp.now(), periods=10, freq="15T")
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        },
        index=timestamps,
    )
    return df

@pytest.mark.order(1)
def test_split_df(df_model):
    train_prop = 60
    val_prop = 10
    test_prop = 30
    rand_state = 42
    # Act
    train_data, val_data, test_data = split_df(df_model, train_prop, val_prop, test_prop, rand_state)
    # Assert
    assert isinstance(train_data, pd.DataFrame)
    assert isinstance(val_data, pd.DataFrame)
    assert isinstance(test_data, pd.DataFrame)
    assert train_data.shape[0] == int(df_model.shape[0] * train_prop / 100)
    assert val_data.shape[0] == int(df_model.shape[0] * val_prop / 100)
    assert test_data.shape[0] == int(df_model.shape[0] * test_prop / 100)

    # Arrange
    train_prop = 0
    val_prop = 30
    test_prop = 70
    # Act
    train_data, val_data, test_data = split_df(df_model, train_prop, val_prop, test_prop, rand_state)
    # Assert
    assert train_data == None
    assert val_data.shape[0] == int(df_model.shape[0] * val_prop / 100)
    assert test_data.shape[0] == int(df_model.shape[0] * test_prop / 100)


@pytest.mark.order(2)
def test_model_definition(df_model):
    train_prop = 60
    val_prop = 10
    test_prop = 30
    rand_state = 42
    train, val, test = split_df(df_model, train_prop, val_prop, test_prop, rand_state)

    n_features = train.shape[1]

    EPOCHS = 2
    BATCH_SIZE = 1

    autoencoder = model_definition(n_features, train, val, EPOCHS, BATCH_SIZE)

    decoded_train = autoencoder_predict(autoencoder, train, "train")
    decoded_test = autoencoder_predict(autoencoder, test, "test")

    assert np.all(decoded_train.index == train.index)
    assert np.all(decoded_test.index == test.index)

    assert decoded_train.shape == train.shape
    assert decoded_test.shape == test.shape

@pytest.mark.order(3)
def test_calculate_threshold():
    # Very simple real-like example
    # in this example if a new observation has an error > than the maximum error from the ND data, it's anomalous
    # therefore the threshold should be equal to the ND maximum error
    MAXIMUM_ERROR = 3
    val_ND = np.array(
        [
            [10, 20],
            [11, 21],
            [12, 20 + MAXIMUM_ERROR],
        ]
    )
    decoded_val_ND = np.array(
        [
            [10, 20],
            [10, 20],
            [10, 20],
        ]
    )
    val_AD = np.array(
        [
            [100, 200],
            [130, 230],
        ]
    )
    decoded_val_AD = np.array(
        [
            [10, 20],  # AD has very high error prediction (decoded_val_AD - val_AD)
            [10, 20],
        ]
    )
    threshold, n_perc = calculate_threshold(val_ND, decoded_val_ND, val_AD, decoded_val_AD)
    assert threshold == MAXIMUM_ERROR

    # Changing only one anomaly point that was predicted with 0 error
    # Useful only for looking at the printed graph
    decoded_val_AD = np.array(
        [
            [10, 20],
            [130, 230],
        ]
    )
    threshold, n_perc = calculate_threshold(val_ND, decoded_val_ND, val_AD, decoded_val_AD)
    assert threshold == MAXIMUM_ERROR

@pytest.mark.order(4)
def test_evaluate_model():
    normal_data = True
    test = np.random.rand(100, 10)
    decoded_test = test
    threshold = 0.5

    classes, precision, recall, fscore = evaluate_model(normal_data, test, decoded_test, threshold)

    assert precision == 1
    assert recall == 1
    assert fscore == 1
