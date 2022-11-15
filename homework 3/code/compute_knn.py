import argparse
import sys
import os
from typing import List, Tuple

import numpy as np
from numba import jit


@jit(nopython=True)
def compute_knn(X, X_tst, k=1):
    N = X.shape[0]
    N_tst = X_tst.shape[0]
    knn_mat = np.zeros((N_tst, k), dtype=np.int64)

    assert N >= k, "k must be less than N"

    for i in range(N_tst):
        knn_mat[i] = np.argsort(np.sum((X - X_tst[i]) ** 2, axis=1))[:k]
        # knn_mat[i] = np.argsort(np.linalg.norm(X - X_tst[i], axis=1))[:k]
    return knn_mat


def predict_labels(y_trn, knn_mat):
    N_tst = knn_mat.shape[0]
    y_pred = np.zeros(N_tst)
    for i in range(N_tst):
        y_pred[i] = np.argmax(np.bincount(y_trn[knn_mat[i]]))
    return y_pred


def accuracy(y_pred: np.ndarray, y_tst: np.ndarray) -> np.ndarray:
    """Compute the accuracy of the predictions
    Args:
        y_pred (np.ndarray): Predicted labels (n, )
        y_tst (np.ndarray): True labels (n, )
    Returns:
        Accuracy
    """
    return np.mean(y_pred == y_tst)


def recall(y_pred: np.ndarray, y_tst: np.ndarray) -> np.ndarray:
    """Compute the recall of the predictions
    Args:
        y_pred (np.ndarray): Predicted labels (n, )
        y_tst (np.ndarray): True labels (n, )
    Returns:
        Recall
    """
    return np.sum(y_pred[y_tst == 1] == 1) / np.sum(y_tst == 1)


def precision(y_pred: np.ndarray, y_tst: np.ndarray) -> np.ndarray:
    """Compute the precision of the predictions
    Args:
        y_pred (np.ndarray): Predicted labels (n, )
        y_tst (np.ndarray): True labels (n, )
    Returns:
        Precision
    """
    return np.sum(y_pred[y_tst == 1] == 1) / np.sum(y_pred == 1)


def split_features_by_lines(data: List[str]) -> List[List[str]]:
    """Split the data by lines. This also removes the header and the string ids
    :param:
        data (list[str]): Data to split
    :return: Data split by lines
    """
    return [i.split('\t')[1:] for i in data][1:]


def convert_to_numpy(data: List[str]) -> np.ndarray:
    """Convert the data to numpy arrays. This function calls the split data by lines function
    to return a clean matrix.
    Args:
        data (list[str]): Data to convert
    Returns:
        Numpy array
"""
    return np.array(split_features_by_lines(data), dtype=np.float32)


def split_labels_by_lines(data):
    """Split the data by lines. This also removes the header and the string ids.
    The labels are converted to integers.
    Args:
        data (list[str]): Data to split
    Returns:
         Data split by lines
    """
    data = [i.split('\t')[1].split('\n')[0] for i in data][1:]
    return [1 if i == "+" else 0 for i in data]


def convert_label_to_numpy(data):
    return np.array(split_labels_by_lines(data), dtype=np.int32)


def preprocess_dataset(train_dir: str, test_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the dataset
    Args:
        train_dir (str): Path to the training data
        test_dir (str): Path to the test data
    """
    DATASET_NAME = ['matrix_mirna_input.txt', 'phenotype.txt']
    # # read a text file with the data
    with open(train_dir + '/' + DATASET_NAME[0], 'r') as f, open(train_dir + '/' + DATASET_NAME[1], 'r') as g:
        x_trn = f.readlines()
        y_trn = g.readlines()

    with open(test_dir + '/' + DATASET_NAME[0], 'r') as f, open(test_dir + '/' + DATASET_NAME[1], 'r') as g:
        x_tst = f.readlines()
        y_tst = g.readlines()

    # split data by lines
    x_trn = convert_to_numpy(x_trn)
    y_trn = convert_label_to_numpy(y_trn)
    x_tst = convert_to_numpy(x_tst)
    y_tst = convert_label_to_numpy(y_tst)

    # print statistics
    print_statistics(x_trn, y_trn, x_tst, y_tst)
    return x_trn, y_trn, x_tst, y_tst


def print_statistics(x_trn, y_trn, x_tst, y_tst):
    print("Training set:")
    print("  Number of samples:", x_trn.shape[0])
    print("  Number of features:", x_trn.shape[1])
    print("  Number of classes:", len(np.unique(y_trn)))
    print("Test set:")
    print("  Number of samples:", x_tst.shape[0])
    print("  Number of features:", x_tst.shape[1])
    print("  Number of classes:", len(np.unique(y_tst)))
    print("  Data type:", x_trn.dtype)
    print("  Data type:", y_trn.dtype)
    print("  Data type:", x_tst.dtype)
    print("  Data type:", y_tst.dtype)


def write_results(metrics: dict, output_dir: str) -> None:
    """Write the results to a file
    Args:
        metrics (dict): Dictionary with the metrics
        output_dir (str): Path to the output directory
    """
    TEXTFILE = '/' + 'output_knn.txt'

    keys = list(metrics.keys())
    print(keys)
    with open(output_dir + TEXTFILE, 'w') as f:
        metric_length = max([len(i) for i in metrics.values()])

        # write the headers
        for key in keys:
            # as long as the key is not the last key
            if key != keys[-1]:
                f.write(key + '\t')
            else:
                f.write(key + '\n')

        for i in range(metric_length):
            for key in keys:
                value = metrics[key][i]
                # as long as the key is not the last key
                if key != keys[-1]:
                    f.write(f"{value:.2f}" + '\t')
                else:
                    f.write(f"{value:.2f}" + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mink", type=int, default=1)
    parser.add_argument("--maxk", type=int, default=99)
    parser.add_argument("--traindir", type=str, default='data/part1/train')
    parser.add_argument("--testdir", type=str, default='data/part1/test')
    parser.add_argument("--outputdir", type=str, default='data/part1/output')
    args = parser.parse_args()

    # preprocess the dataset
    x_trn, y_trn, x_tst, y_tst = preprocess_dataset(args.traindir, args.testdir)

    ks = np.arange(args.mink, args.maxk + 1, 1)
    compute_knn(x_trn, x_tst)  # numba warmup
    metrics = {'Value of k': [], 'accuracy': [], 'recall': [], 'precision': []}
    for k in ks:
        knn_mat = compute_knn(x_trn, x_tst, k)
        y_pred = predict_labels(y_trn, knn_mat)
        metrics['accuracy'].append(accuracy(y_pred, y_tst))
        metrics['recall'].append(recall(y_pred, y_tst))
        metrics['precision'].append(precision(y_pred, y_tst))
        metrics['Value of k'].append(int(k))

    # check if directory exists, otherwise create it
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    # save the results
    write_results(metrics, args.outputdir)


if __name__ == '__main__':
    main()
