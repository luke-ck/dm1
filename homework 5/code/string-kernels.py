import numpy as np


def compute_kbase_kernel(S, T):
    if S[0] == 'G' or T[0] == 'G':
        return 0
    else:
        return np.sum([1 for i in range(len(S)) if S[i] == T[i]])


def compute_rkernel(X, Y):
    # kbase(s, s′) = {0 if s_0 = G or s′_0 = G else ∑_{i = 0}^2 k(s_i, s′_i)
    kbase = 0
    for i in range(len(X) - 2):
        for j in range(len(Y) - 2):
            kbase += compute_kbase_kernel(X[i:i + 3], Y[j:j + 3])
            # if kbase != 0:
            #     print(X[i:i + 3], Y[j:j + 3], kbase)

    return kbase


if __name__ == '__main__':
    # Define the two strings
    X1 = 'GPAGFAGPPGDA'
    X2 = 'PRGDQGPVGRTG'
    X3 = 'GFPNFDVSVSDM'

    # Compute the kernel
    print(compute_rkernel(X1, X2))
    print(compute_rkernel(X1, X3))