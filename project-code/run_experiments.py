"""
Experiments on the toy HMM. hmm.py holds the algorithms; this file runs them and saves the results.

One function per experiment. Each saves a JSON file in results/ (see utils.py); plot_results.py reads
those files. To refresh a figure after changing values: run this file, then plot_results.py.

Run:  ../.venv/bin/python run_experiments.py
"""
import numpy as np

from hmm import STATES, SYMBOLS, pi, A, B, sample_hmm_sequence, random_parameters, forward_algorithm, baum_welch
from utils import save_results


def run_baum_welch_restarts(T=2000, n_restarts=5, max_iterations=200):
    """
    Baum-Welch from several random starts on one long toy sequence.

    Parameters
    ----------
    T: int, length of the sampled observation sequence (default 2000)
    n_restarts: int, number of random starting points (default 5); restart k uses random seed k
    max_iterations: int, upper limit on EM iterations per start (default 200)

    Returns
    -------
    None; writes results/baum_welch_restarts.json with the settings, state and symbol names,
    the true pi, A, B and their log-likelihood, and per restart its number, fitted pi, A, B and log-likelihoods
    """
    # 1. one long observation sequence from the true model
    print(f"Baum-Welch restarts: T={T}, n_restarts={n_restarts}, max_iterations={max_iterations}")
    _, observations = sample_hmm_sequence(pi, A, B, T, np.random.default_rng(0))
    # 2. log-likelihood under the true parameters, the reference value
    _, _, log_likelihood_true = forward_algorithm(pi, A, B, observations)
    print(f"log-likelihood under the true parameters: {log_likelihood_true:.1f}")

    # 3. Baum-Welch from n_restarts random starts; restart k is seeded with k so it can be reproduced
    restarts = []
    for restart in range(1, n_restarts + 1):
        pi_start, A_start, B_start = random_parameters(len(STATES), len(SYMBOLS), np.random.default_rng(seed=restart))
        pi_fit, A_fit, B_fit, log_likelihoods = baum_welch(pi_start, A_start, B_start, observations, max_iterations)
        assert np.all(np.diff(log_likelihoods) >= -1e-8), "log-likelihood decreased during Baum-Welch"
        restarts.append({"restart": restart, "pi": pi_fit, "A": A_fit, "B": B_fit, "log_likelihoods": log_likelihoods})
        print(f"restart {restart}: {len(log_likelihoods)} iterations, "
              f"log-likelihood {log_likelihoods[0]:.1f} -> {log_likelihoods[-1]:.1f}")

    # 4. save everything the plots and the report need
    path = save_results("baum_welch_restarts", {
        "T": T, "n_restarts": n_restarts, "max_iterations": max_iterations,
        "states": STATES, "symbols": SYMBOLS,
        "true": {"pi": pi, "A": A, "B": B, "log_likelihood": log_likelihood_true},
        "restarts": restarts,
    })
    best = max(restarts, key=lambda r: r["log_likelihoods"][-1])
    print(f"best restart: {best['restart']} (log-likelihood {best['log_likelihoods'][-1]:.1f})")
    print("saved", path)


if __name__ == "__main__":
    run_baum_welch_restarts()
