import argparse
from typing import Tuple, List
import pandas as pd
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series

CTR = 0  # used to differentiate what label to write to file


def compute_frequency_per_feature(mat: np.ndarray, col: int) -> np.ndarray:
    """
    Compute the frequency of each feature. Since the features are categorical
    between 1 and 10, we can use a bincount to compute the frequency of each feature.
    Args:
        mat (np.ndarray): Training data (n, d)
        col (int): Column index
    Returns:
        freq (np.ndarray): Frequency of each feature in the column
    """
    return np.bincount(mat[:, col].astype(np.int64))[1:11]


def split_by_label(features, labels):
    X_0 = features[labels == 2]
    X_1 = features[labels == 4]

    return X_0, X_1


def compute_frequency_counts_features(features: np.ndarray, sizes) -> np.ndarray:
    """
    Compute the frequency of each feature in each class.
    Args:
        features (np.ndarray): Training data (n, d)
    Returns:
        freq (np.ndarray): Frequency of each feature in each class (d, 10)
    """
    samples = features.shape[0]
    no_features = features.shape[1]
    counts = np.zeros((no_features, 10))
    for col in range(no_features):
        freq = compute_frequency_per_feature(features, col)
        if freq.shape[0] < 10:
            freq = np.append(freq, np.zeros(10 - freq.shape[0]))
        counts[col] = freq / (samples - sizes[col])
    return counts


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into features and labels. Labels are returned as a numpy array.
    Args:
        data (list): List of strings
    """

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return x, y


def count_nans(x: List[pd.DataFrame]) -> List[pd.Series]:
    """Count the number of NaNs in each column
    Args:
        x (List[pd.DataFrame]): List of dataframes
    Returns:
        counts (List[pd.Series]): List of series containing the counts
    """
    counts = []
    for i in range(len(x)):
        counts.append(x[i].isna().sum())
    return counts


def preprocess_dataset(data_dir: str, test: bool = False) -> Tuple[List[ndarray | ndarray], ndarray, List[Series]]:
    """Preprocess the dataset
    Args:
        data_dir (str): Path to the dataset directory
        test (bool): Whether the dataset is the test set
    Returns:
        x (np.ndarray): features (n, d)
        y (np.ndarray): labels (n, )
    """
    TEXTFILE = '/' + 'tumor_info.txt'

    data = pd.read_csv(data_dir + TEXTFILE, sep='\t', header=None)
    x, y = split_data(data)
    if test:
        no_nans = count_nans([x])
        print("Number of NaNs in test set: {}".format(no_nans[0]))
        x = handle_nans([x], test=True)
        return x, y.to_numpy().astype(np.int8), no_nans

    train_sets = split_by_class(x, y)
    no_nans = count_nans(train_sets)
    train_sets = handle_nans(train_sets)

    return train_sets, y.to_numpy().astype(np.int8), no_nans


def handle_nans(x: List[pd.DataFrame], test: bool = False) -> List[np.ndarray]:
    """Handle NaNs in the dataset. Return a list of numpy arrays.
    Args:
        x (List[pd.DataFrame]): List of dataframes
    Returns:
        x (List[pd.DataFrame]): List of dataframes
        test (bool): Whether the dataset is the test set
    """
    if test:
        return [x[0].to_numpy().astype(np.int8)]

    train_sets = []
    for i in range(len(x)):
        train_sets.append(x[i].copy().replace(np.nan, 11).to_numpy())  # hacky at best. replace nan with 11 since
        # we know that the values are between 1 and 10
    return train_sets


def split_by_class(x: pd.DataFrame, y: pd.DataFrame) -> List[pd.DataFrame]:
    classes = np.unique(y)
    features_under_class = []
    for i in range(len(classes)):
        features_under_class.append(x[y == classes[i]])
    return features_under_class


def make_prediction(freqs: List[np.ndarray | np.ndarray], priors: List[float], x_tst: np.ndarray) -> int:
    """
    Make a prediction for each sample in the test set.
    Args:
        freqs (List[np.ndarray | np.ndarray]): List of feature frequencies
        priors (List[float]): List of priors
        x_tst (np.ndarray): Test data (d, )
    Returns:
        pred (int): Predicted class
    """
    p0max = 0
    p1max = 0
    freq_0, freq_1 = freqs
    prior_y0, prior_y1 = priors

    for i in range(x_tst.shape[0]):
        p0 = freq_0[i, x_tst[i] - 1]
        p1 = freq_1[i, x_tst[i] - 1]
        if p0 == 0:
            p0 = 1e-10  # avoid zero probabilities
        if p1 == 0:
            p1 = 1e-10
        p0max += np.log(p0)
        p1max += np.log(p1)
    p0max += np.log(prior_y0)
    p1max += np.log(prior_y1)

    if p0max > p1max:
        pred = 2
    else:
        pred = 4
    return pred


def write_output(outdir, counts):
    global CTR

    assert CTR < 2, "There should only be two labels to write to file"
    CTR += 1
    values = np.arange(1, 11, 1)[np.newaxis, :]
    output = np.concatenate((values, counts), axis=0).T
    COLUMNS = ["value", "clump", "uniformity", "marginal", "mitoses"]
    TEXTFILE = '/output_summary_class_'
    # write to file

    label = 2 if CTR % 2 != 0 else 4
    df = pd.DataFrame(output, columns=COLUMNS).set_index("value")
    df.index = df.index.astype(int)
    df.to_csv(outdir + TEXTFILE + str(label) + '.txt', sep='\t', float_format='%.3f')


def make_predictions(counts, priors, x_tst):
    """
    Make predictions for the test set.
    Args:
        counts (List[np.ndarray | np.ndarray]): List of feature frequencies
        priors (List[float]): List of priors
        x_tst (np.ndarray): Test data (n, d)
    Returns:
        preds (np.ndarray): Predictions (n, )
    """
    preds = np.zeros(x_tst.shape[0], dtype=np.int8)
    for i in range(x_tst.shape[0]):
        preds[i] = make_prediction(counts, priors, x_tst[i])
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindir", type=str, default='data/part2/train', required=True)
    parser.add_argument("--testdir", type=str, default='data/part2/test', required=False)
    parser.add_argument("--outdir", type=str, default='data/part2/output', required=True)

    args = parser.parse_args()

    # preprocess the dataset
    x_trn, y_trn, no_nans = preprocess_dataset(args.traindir)

    assert len(x_trn) == 2, "There should be two classes in the training set."
    # split the data into two classes
    priors = []
    sizeof_x_trn = 0
    for i in range(len(x_trn)):
        sizeof_x_trn += x_trn[i].shape[0]
    for i in range(len(x_trn)):
        priors.append(len(x_trn[i]) / sizeof_x_trn)
    # compute the frequency of each feature in each class
    freqs = []
    for i, counts in enumerate(no_nans):
        freqs.append(compute_frequency_counts_features(x_trn[i], counts))

    for freq in freqs:
        # write the output to a file
        write_output(args.outdir, freq)

    x_tst = np.array([5, 2, 3, 1])
    # make predictions on the test set
    pred = make_prediction(freqs, priors, x_tst)
    print("Sample test point prediction:", pred)
    if args.testdir:
        x_tst, y_tst, _ = preprocess_dataset(args.testdir, test=True)
        # make predictions on the test set
        preds = make_predictions(freqs, priors, x_tst[0].astype(np.int8))
        print("Predictions: ", preds)


if __name__ == "__main__":
    main()
