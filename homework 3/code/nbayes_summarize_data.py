import argparse
from typing import Tuple
import pandas as pd
import numpy as np


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
    return np.bincount(mat[:, col].astype(np.int64))[1:]


def split_by_label(features, labels):
    X_0 = features[labels == 2]
    X_1 = features[labels == 4]

    return X_0, X_1


def compute_frequency_counts_features(features: np.ndarray) -> np.ndarray:
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
        counts[col] = freq / samples
    return counts


def split_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Split the data into features and labels
    Args:
        data (list): List of strings
    """
    data.dropna(inplace=True)  # drop rows with missing values
    x = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    return x, y


def preprocess_dataset(data_dir):
    """Preprocess the dataset
    Args:
        data_dir (str): Path to the dataset directory
    Returns:
        x (np.ndarray): features (n, d)
        y (np.ndarray): labels (n, )
    """
    TEXTFILE = '/' + 'tumor_info.txt'

    data = pd.read_csv(data_dir + TEXTFILE, sep='\t', header=None)

    # split into features and labels
    x, y = split_data(data)
    return x.astype(np.int64), y.astype(np.int8)


def make_prediction(freq_0: np.ndarray, freq_1: np.ndarray, prior_y0: float, prior_y1: float, x_tst: np.ndarray) -> int:
    """
    Make a prediction for each sample in the test set.
    Args:
        freq_0 (np.ndarray): Probability P(x_i | y=0)
        freq_1 (np.ndarray): Probability P(x_i | y=1)
        prior_y0 (float): Prior probability of class 0
        prior_y1 (float): Prior probability of class 1
        x_tst (np.ndarray): Test data (d, )
    Returns:
        pred (int): Predicted class
    """
    p0max = 0
    p1max = 0
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


def write_output(outdir, counts_0, counts_1):
    values = np.arange(1, 11, 1)[np.newaxis, :]
    output_0 = np.concatenate((values, counts_0), axis=0).T
    output_1 = np.concatenate((values, counts_1), axis=0).T
    output = [output_0, output_1]
    COLUMNS = ["value", "clump", "uniformity", "marginal", "mitoses"]
    TEXTFILE = '/output_summary_class_'
    # write to file
    for i in range(2):
        label = 2 if i == 0 else 4
        df = pd.DataFrame(output[i], columns=COLUMNS).set_index("value")
        df.index = df.index.astype(int)
        df.to_csv(outdir + TEXTFILE + str(label) + '.txt', sep='\t', float_format='%.3f')


def make_predictions(counts_0, counts_1, prior_y0, prior_y1, x_tst):
    """
    Make predictions for each sample in the test set.
    Args:
        counts_0 (np.ndarray): Probability P(x_i | y=0)
        counts_1 (np.ndarray): Probability P(x_i | y=1)
        prior_y0 (float): Prior probability of class 0
        prior_y1 (float): Prior probability of class 1
        x_tst (np.ndarray): Test data (n, d)
    Returns:
        preds (np.ndarray): Predictions for each sample in the test set (n, )
    """
    predictions = []
    for i in range(x_tst.shape[0]):
        pred = make_prediction(counts_0, counts_1, prior_y0, prior_y1, x_tst[i])
        predictions.append(pred)
    return np.array(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindir", type=str, default='data/part2/train', required=True)
    parser.add_argument("--testdir", type=str, default='data/part2/test', required=False)
    parser.add_argument("--outdir", type=str, default='data/part2/output', required=True)

    args = parser.parse_args()

    # preprocess the dataset
    x_trn, y_trn = preprocess_dataset(args.traindir)

    # split the data into two classes
    X_0, X_1 = split_by_label(x_trn, y_trn)
    prior_y0 = len(X_0) / len(x_trn)
    prior_y1 = len(X_1) / len(x_trn)
    # compute the frequency of each feature in each class
    counts_0 = compute_frequency_counts_features(X_0)
    counts_1 = compute_frequency_counts_features(X_1)

    # write the output to a file
    write_output(args.outdir, counts_0, counts_1)

    x_tst = np.array([5, 2, 3, 1])
    # make predictions on the test set
    pred = make_prediction(counts_0, counts_1, prior_y0, prior_y1, x_tst)
    print(pred)
    if args.testdir:
        x_tst, y_tst = preprocess_dataset(args.testdir)
        # make predictions on the test set
        pred = make_predictions(counts_0, counts_1, prior_y0, prior_y1, x_tst)
        print(pred)


if __name__ == "__main__":
    main()
