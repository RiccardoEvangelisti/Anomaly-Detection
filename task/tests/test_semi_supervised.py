import numpy as np
import pandas as pd

from task.semi_supervised.utils import calculate_threshold, split_df


def test_split_df():
    # Arrange
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5,6,7,8,9,10], "col2": ["a", "b", "c", "d", "e","f","g", "h", "i", "j"]})
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
    # Arrange
    val_ND = np.random.rand(10, 10)
    decoded_val_ND = np.random.rand(10, 10)
    val_AD = np.random.rand(10, 10)
    decoded_val_AD = np.random.rand(10, 10)
    threshold = calculate_threshold(val_ND, decoded_val_ND, val_AD, decoded_val_AD)
    assert isinstance(threshold, float)