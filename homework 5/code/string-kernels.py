import numpy as np

k = 3


def dirac_kernel(X, Y):
    if X == Y:
        return 1
    else:
        return 0


def compute_kbase_kernel(S, T):
    assert len(S) == k
    assert len(T) == k

    if S[0] != 'G' or T[0] != 'G':
        return 0
    else:
        return np.sum([dirac_kernel(S[i], T[i]) for i in range(len(S))])


def compute_rkernel(X: str, Y: str, k=3) -> float:
    # kbase(s, s′) = {0 if s_0 != G or s′_0 != G else ∑_{i = 0}^2 k(s_i, s′_i)
    kbase = 0
    X_kmers = []
    Y_kmers = []
    for i in range(len(X) - k + 1):
        for j in range(len(Y) - k + 1):
            X_kmer = X[i:i + k]
            Y_kmer = Y[j:j + k]

            if X_kmer not in X_kmers:
                X_kmers.append(X_kmer)
            if Y_kmer not in Y_kmers:
                Y_kmers.append(Y_kmer)

            kbase += compute_kbase_kernel(X_kmer, Y_kmer)

    print(f" string {X.__repr__()} k_mers => {X_kmers}")
    print(f" string {Y.__repr__()} k_mers => {Y_kmers}")

    return kbase


if __name__ == '__main__':
    # Define the two strings
    X1 = 'GPAGFAGPPGDA'
    X2 = 'PRGDQGPVGRTG'
    X3 = 'GFPNFDVSVSDM'
    # Compute the kernel
    print(f"k(X1,X2)={compute_rkernel(X1, X2, k)}")
    print(f"k(X1,X3)={compute_rkernel(X1, X3, k)}")