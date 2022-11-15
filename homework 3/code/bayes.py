import argparse


def main():
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nmax", type=int, default=1)
    parser.add_argument("--D", type=int, default=1)
    args = parser.parse_args()
    Nmax = args.Nmax
    D = args.D

    uniform_prior = 1 / Nmax

    evidence = 0
    for i in range(D, Nmax + 1):
        evidence += (1 / i) * uniform_prior

    max_posterior = 0
    expected_value = 0
    max_N = 0
    # calculate for which N the posterior is the highest
    for N in range(D, Nmax + 1):
        likelihood = 1 / N
        posterior = likelihood * uniform_prior / evidence
        expected_value += posterior * N
        if posterior > max_posterior:
            max_posterior = posterior
            max_N = N

    print(f"Maximum posterior: {max_posterior:.6f} achieved for N = {max_N}")
    print(f"Expected value: {expected_value:.6f}")

if __name__ == '__main__':
    main()
