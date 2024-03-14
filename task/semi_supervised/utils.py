import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def split_df(df, train=60, val=10, test=20, rand=None):
    if train + val + test != 100:
        raise ValueError("The sum must be 100")
    temp, test_data = train_test_split(df, test_size=test / 100.0, random_state=rand)
    train_data, val_data = (
        (None, temp) if train == 0 else train_test_split(temp, test_size=val / (100 - test), random_state=rand)
    )
    return train_data, val_data, test_data


def calculate_threshold(val_ND, decoded_val_ND, val_AD, decoded_val_AD):

    # A list in which each element corresponds to the maximum absolute error of one row. **It's the maximum error that an observation has made**
    max_errors_list_valid_ND = np.max(np.abs(decoded_val_ND - val_ND), axis=1)
    max_errors_list_valid_AD = np.max(np.abs(decoded_val_AD - val_AD), axis=1)

    classes = np.concatenate(([0] * val_ND.shape[0], [1] * val_AD.shape[0]))
    errors = np.concatenate((max_errors_list_valid_ND, max_errors_list_valid_AD))

    n_perc_min = 70
    n_perc_max = 100
    best_n_perc = n_perc_max
    fscore_val_best = 0
    fscore_val_AD_best = 0
    fscore_val_ND_best = 0
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

        precision_val_ND, recall_val_ND, fscore_val_ND, _ = precision_recall_fscore_support(
            classes, predictions, average="binary", pos_label=0
        )
        precision_val_AD, recall_val_AD, fscore_val_AD, _ = precision_recall_fscore_support(
            classes, predictions, average="binary", pos_label=1
        )
        precision_val, recall_val, fscore_val, _ = precision_recall_fscore_support(
            classes, predictions, average="weighted"
        )

        fscores.append(fscore_val)
        precs.append(precision_val)
        recalls.append(recall_val)
        n_percs.append(n_perc)

        if fscore_val > fscore_val_best:
            precision_val_best = precision_val
            precision_ND_best = precision_val_ND
            precision_A_best = precision_val_AD
            recall_val_best = recall_val
            recall_val_ND_best = recall_val_ND
            recall_val_AD_best = recall_val_AD
            fscore_val_best = fscore_val
            fscore_val_ND_best = fscore_val_ND
            fscore_val_AD_best = fscore_val_AD
            best_n_perc = n_perc

    best_error_threshold = np.percentile(max_errors_list_valid_ND, best_n_perc)
    print("Best threshold: {}".format(best_error_threshold))

    """
    21) in another function, called like test_autoencoder(), where it will be passed test_AD and test_ND: """

    return best_error_threshold, n_perc
