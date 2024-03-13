from keras import regularizers
from keras.layers import Dense, Input
from keras.models import Model

import matplotlib.pyplot as plt

from utils import calculate_threshold, split_df

RANDOM_STATE = 42
TRAIN_ND_PERC, VAL_ND_PERC, TEST_ND_PERC = 60, 10, 30
VAL_AD_PERC, TEST_AD_PERC = 30, 70

EPOCHS = 256
BATCH_SIZE = 128


def main():
    """- Build the complete dataset, with anomalies and with nagios column"""

    df = build_dataset()

    n_features = df.shape[1]

    """ Extract dynamically the anomaly periods, as list of time intervals
        # Based on the nagios signal: if equals to 1 is anomaly period."""
    df_ND, df_AD = extract_anomalous_data(df)

    train_ND, val_ND, test_ND = split_df(
        df_ND, train=TRAIN_ND_PERC, val=VAL_ND_PERC, test=TEST_ND_PERC, rand=RANDOM_STATE
    )

    # Autoencoder definition
    input_data = Input(shape=(n_features,))
    encoded = Dense(n_features * 10, activation="relu", activity_regularizer=regularizers.l1(1e-5))(input_data)
    decoded = Dense(n_features, activation="linear")(encoded)

    autoencoder = Model(input_data, decoded)
    autoencoder.compile(optimizer="adam", loss="mean_absolute_error")
    history = autoencoder.fit(
        train_ND,
        train_ND,  # It's using train_ND as both the input and the target since this is a reconstruction model
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(val_ND, val_ND),
        # sample_weight=np.asarray(active_idle), # not used for now
        verbose=1,
    )

    plt.plot(autoencoder.history["loss"], label="Training Loss")
    plt.plot(autoencoder.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

    # Prediction
    decoded_val_ND = autoencoder.predict(val_ND)
    decoded_AD = autoencoder.predict(df_AD)

    # Split AD predictions
    _, val_AD, test_AD = split_df(df_AD, train=0, val=VAL_AD_PERC, test=TEST_AD_PERC, rand=RANDOM_STATE)
    _, decoded_val_AD, decoded_test_AD = split_df(
        decoded_AD, train=0, val=VAL_AD_PERC, test=TEST_AD_PERC, rand=RANDOM_STATE
    )

    # Find best Threshold
    threshold = calculate_threshold(val_ND, decoded_val_ND, val_AD, decoded_val_AD)

    """
    20) final test: 
        - test on test_AD
        - test on (unseen) test_ND
    """


if __name__ == "__main__":
    main()
