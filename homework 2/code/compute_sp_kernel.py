import argparse
import os
import sys
from scipy.io import loadmat
from shortest_path_kernel import floyd_warshall, sp_kernel
import numpy as np
import time


def compute_sp_kernel(A, B):
    K = []
    for i in range(len(A)):
        for j in range(len(B)):
            K.append(sp_kernel(A[i], B[j]))
    K = np.array(K, dtype=np.float64)
    return np.mean(K)


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken by {func.__name__} is {end - start} seconds")
        return result

    return wrapper


if __name__ == '__main__':
    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute distance functions on time-series"
    )
    parser.add_argument(
        "--datadir",
        required=True,
        help="Path to input directory containing file EGC200_TRAIN.txt"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where timeseries_output.txt will be created"
    )

    args = parser.parse_args()

    # Set the paths
    data_dir = args.datadir
    out_dir = args.outdir

    os.makedirs(args.outdir, exist_ok=True)

    # Read the file
    mat = loadmat("{}/{}".format(args.datadir, 'MUTAG.mat'))

    label = np.reshape(mat['lmutag'], (len(mat['lmutag'], )))
    data = np.reshape(mat['MUTAG']['am'], (len(label),))

    # Compute shortest-path matrices
    S = np.array([floyd_warshall(x) for x in data], dtype=object)

    # split S by label
    S1 = S[label == 1]
    S2 = S[label == -1]
    S = [S1, S2]
    sp_kernel(S1[0], S2[0])  # warm up to compile numba functions
    # Create the output file
    try:
        file_name = "{}/graphs_output.txt".format(args.outdir)
        f_out = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)

    lst_group = ['mutagenic', 'non-mutagenic']

    # Write header for output file
    f_out.write('{}\t{}\n'.format(
        'Pair of classes',
        'SP'))

    # compute the average shortest-path kernel
    for i in range(len(S)):
        for j in range(i, len(S)):
            res = timeit(compute_sp_kernel)(S[i], S[j])
            f_out.write('{}:{}\t{}\n'.format(lst_group[i], lst_group[j], res))
    sys.exit(0)
