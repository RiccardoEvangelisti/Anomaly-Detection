import numpy as np
import pandas as pd

from task.semi_supervised.utils import calculate_threshold, split_df


def test_split_df():
    # Arrange
    df = pd.DataFrame(
        {"col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "col2": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]}
    )
    train_prop = 60
    val_prop = 10
    test_prop = 30
    rand_state = 42
    # Act
    train_data, val_data, test_data = split_df(df, train_prop, val_prop, test_prop, rand_state)
    # Assert
    assert train_data.shape[0] == int(df.shape[0] * train_prop / 100)
    assert val_data.shape[0] == int(df.shape[0] * val_prop / 100)
    assert test_data.shape[0] == int(df.shape[0] * test_prop / 100)

    # Arrange
    train_prop = 0
    val_prop = 30
    test_prop = 70
    # Act
    train_data, val_data, test_data = split_df(df, train_prop, val_prop, test_prop, rand_state)
    # Assert
    assert train_data == None
    assert val_data.shape[0] == int(df.shape[0] * val_prop / 100)
    assert test_data.shape[0] == int(df.shape[0] * test_prop / 100)


def test_calculate_threshold():
    # All equal values
    val_ND = np.array([[0, 1], [2, 3], [4, 5]])
    decoded_val_ND = np.array([[0, 1], [2, 3], [4, 5]])
    val_AD = np.array([[0, 1], [2, 3]])
    decoded_val_AD = np.array([[0, 1], [2, 3]])
    threshold, n_perc = calculate_threshold(val_ND, decoded_val_ND, val_AD, decoded_val_AD)
    assert isinstance(threshold, float)
    assert n_perc <= 100
    assert threshold == 0

    # Real-like example
    # if a new observation has an error > than the maximum error from the ND data, it's anomalous
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
            [10, 20], # AD has very high error prediction (decoded_val_AD - val_AD)
            [10, 20],
        ]
    )
    threshold, n_perc = calculate_threshold(val_ND, decoded_val_ND, val_AD, decoded_val_AD)
    assert threshold == MAXIMUM_ERROR