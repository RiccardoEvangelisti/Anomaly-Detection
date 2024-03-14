import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from utils import autoencoder_predict, calculate_threshold, evaluate_model, model_definition, split_df

"""
ND: Normal Data
AD: Anomalous Data
"""

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

    # Split ND data
    train_ND, val_ND, test_ND = split_df(
        df_ND,
        train=TRAIN_ND_PERC,
        val=VAL_ND_PERC,
        test=TEST_ND_PERC,
        rand=RANDOM_STATE,
    )

    # Autoencoder definition
    autoencoder = model_definition(n_features, train_ND, val_ND, EPOCHS, BATCH_SIZE)

    plt.plot(autoencoder.history["loss"], label="Training Loss")
    plt.plot(autoencoder.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

    # Prediction of normal data
    _ = autoencoder_predict(autoencoder, train_ND, "ND train")
    decoded_test_ND = autoencoder_predict(autoencoder, test_ND, "ND test")
    decoded_val_ND = autoencoder_predict(autoencoder, val_ND, "ND val")

    # Prediction of anomalous data
    decoded_AD = autoencoder_predict(autoencoder, df_AD, "AD")

    # Split AD data and predictions
    _, val_AD, test_AD = split_df(
        df_AD,
        train=0,
        val=VAL_AD_PERC,
        test=TEST_AD_PERC,
        rand=RANDOM_STATE,
    )
    _, decoded_val_AD, decoded_test_AD = split_df(
        decoded_AD,
        train=0,
        val=VAL_AD_PERC,
        test=TEST_AD_PERC,
        rand=RANDOM_STATE,
    )

    # Find best Threshold using validation sets
    threshold, _ = calculate_threshold(
        val_ND,
        decoded_val_ND,
        val_AD,
        decoded_val_AD,
    )

    # Test on unseen data
    classes_test_ND, precision_test_ND, recall_test_ND, fscore_test_ND = evaluate_model(True, test_ND, decoded_test_ND, threshold)
    classes_test_AD, precision_test_AD, recall_test_AD, fscore_test_AD = evaluate_model(False, test_AD, decoded_test_AD, threshold)

    print("ND TEST: precision = {} recall = {} fscore = {}".format(precision_test_ND, recall_test_ND, fscore_test_ND))
    print("AD TEST: precision = {} recall = {} fscore = {}".format(precision_test_AD, recall_test_AD, fscore_test_AD))
    
    plt.plot(classes_test_ND, label="test ND")
    plt.plot(classes_test_AD, label="test AD")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
