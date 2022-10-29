"""Skeleton file for your solution to the shortest-path kernel."""
import numpy as np
# from numba import jit


def prepare_adj_matrix(A) -> np.ndarray[np.uint16]:
    A[A == 0] = np.iinfo(np.uint8).max  # set all 0s to max uint8
    # the matrix has precision uint16 because we need to make sure that
    # when summing over possibly max values (which are uint8), we don't overflow
    # this is the case when A.shape[0] < 257 (because 257 * 255 = 65280) and max uint16 is 65535
    np.fill_diagonal(A, 0)
    return A


def floyd_warshall(A) -> np.ndarray[np.uint16]:
    """Implement the Floyd--Warshall on an adjacency matrix A.

    Parameters
    ----------
    A : `np.array` of shape (n, n)
        Adjacency matrix of an input graph. If A[i, j] is `1`, an edge
        connects nodes `i` and `j`.

    Returns
    -------
    An `np.array` of shape (n, n), corresponding to the shortest-path
    matrix obtained from A.
    """

    D = prepare_adj_matrix(A.astype(np.uint16))
    n = A.shape[0]
    for k in range(n):
        for i in range(n):
            D[i, :] = np.minimum(D[i, :], D[i, k] + D[k, :])
    return D


# @jit(nopython=True)
def sp_kernel(S1, S2) -> float:
    """Calculate shortest-path kernel from two shortest-path matrices.

    Parameters
    ----------
    S1: `np.array` of shape (n, n)
        Shortest-path matrix of the first input graph.

    S2: `np.array` of shape (m, m)
        Shortest-path matrix of the second input graph.

    Returns
    -------
    A single `float`, corresponding to the kernel value of the two
    shortest-path matrices
    """

    n = S1.shape[0]
    m = S2.shape[0]
    K = 0

    for i in range(n):
        for j in range(i+1, n):
            for k in range(m):
                for l in range(k+1, m):
                    K += (S1[i, j] == S2[k, l])

    return float(K)
