import argparse
import sys

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


def accuracy(y_pred, y_tst):
    """Compute the accuracy of the predictions
    Args:
        y_pred (np.ndarray): Predicted labels (n, )
        y_tst (np.ndarray): True labels (n, )
    """
    return np.mean(y_pred == y_tst)


def split_features_by_lines(data: list[str]) -> list[list[str]]:
    """Split the data by lines. This also removes the header and the string ids
    :param:
        data (list[str]): Data to split
    :return: Data split by lines
    """
    return [i.split('\t')[1:] for i in data][1:]


def convert_to_numpy(data):
    """Convert the data to numpy arrays. This function calls the split data by lines function
    to return a clean matrix.
    :param:
        data (list[str]): Data to convert
    :return: Numpy array
"""
    return np.array(split_features_by_lines(data), dtype=np.float32)


def split_labels_by_lines(data):
    """Split the data by lines. This also removes the header and the string ids.
    The labels are converted to integers.
    :param:
        data (list[str]): Data to split
    :return: Data split by lines
    """
    data = [i.split('\t')[1].split('\n')[0] for i in data][1:]
    return [1 if i == "+" else 0 for i in data]


def convert_label_to_numpy(data):
    return np.array(split_labels_by_lines(data), dtype=np.int32)


def preprocess_dataset(train_dir, test_dir):
    """Preprocess the dataset
    Args:
        data_dir (str): Path to the dataset directory
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mink", type=int, default=1)
    parser.add_argument("--maxk", type=int, default=99)
    parser.add_argument("--traindir", type=str, default='data/part1/train')
    parser.add_argument("--testdir", type=str, default='data/part1/test')

    args = parser.parse_args()

    # preprocess the dataset
    x_trn, y_trn, x_tst, y_tst = preprocess_dataset(args.traindir, args.testdir)

    ks = np.arange(args.mink, args.maxk + 1, 2)
    knn_mat = compute_knn(x_trn, x_tst)  # numba warmup

    for k in ks:
        knn_mat = compute_knn(x_trn, x_tst, k)
        y_pred = predict_labels(y_trn, knn_mat)
        acc = accuracy(y_pred, y_tst)
        print('k = {}, acc = {}'.format(k, acc))



    # knn_mat = compute_knn(x_trn, x_tst, k=4)
    # print("accuracy", accuracy(predict_labels(y_trn, knn_mat), y_tst))
    # knn_mat = compute_knn(x_trn, x_tst, k=1)
    # print("accuracy", accuracy(predict_labels(y_trn, knn_mat), y_tst))

if __name__ == '__main__':
    main()
