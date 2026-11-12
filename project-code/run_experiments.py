"""
Experiments on the toy HMM and the toy population model. hmm.py, population_model.py and
starting_values.py hold the algorithms; this file runs them and saves the results.

One function per experiment. Each saves a JSON file in results/ (see utils.py); plot_results.py reads
those files. To refresh a figure after changing values: run this file, then plot_results.py.

Run:  ../.venv/bin/python run_experiments.py
"""
import numpy as np

import population_model
import starting_values
import toy_parameters as toy   # toy.pi, toy.A, ...; hmm.py below has its own pi, A, STATES

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
        pi_start, A_start, B_start = random_parameters(
            len(STATES),
            len(SYMBOLS),
            np.random.default_rng(seed=restart),
        )
        pi_fit, A_fit, B_fit, log_likelihoods = baum_welch(
            pi_start,
            A_start,
            B_start,
            observations,
            max_iterations,
        )
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


def run_population_recovery(cells_per_week, number_of_random_starts=20, max_iterations=2000, seed=0):
    """
    Toy population model: simulate cells from the true parameters, fit them from the
    Householder start and several random starts, and compare the best fit with the truth.
    Expected: the MA row of A recovered well; the T and MU rows worse at 300
    cells per week, because few cells are in T and MU in the early weeks.

    Parameters
    ----------
    cells_per_week:          int, number of simulated cells in each week (300 and 3000 in the report)
    number_of_random_starts: int, random starts besides the Householder start (default 20; with 5, the
                             3000-cell fit missed the best l: only about 1 random start in 4 reaches it)
    max_iterations:          int, upper limit on EM iterations per start (default 2000)
    seed:                    int, seeds the simulated cells and all starting values

    Returns
    -------
    None; writes results/population_recovery_<cells_per_week>.json with the settings, the simulated
    cells (X, weeks, true states), the true parameters with their p_t and l, every fit with its states
    matched to MA, T, MU, the index of the best fit, and the best fit's p_t and observed soft proportions
    """
    print(f"\nPopulation recovery: {cells_per_week} cells per week, Householder start + "
          f"{number_of_random_starts} random starts, max_iterations={max_iterations}")
    random_generator = np.random.default_rng(seed)

    # 1. Simulated cells from the true parameters; states is the answer key, never given to the fit
    X, weeks, states = population_model.sample_cells(
        toy.pi,
        toy.A,
        toy.means,
        toy.variances,
        cells_per_week,
        toy.NUMBER_OF_WEEKS,
        random_generator,
    )

    # 2. l under the true parameters, the reference value
    _, log_likelihood_true = population_model.e_step(
        X,
        weeks,
        toy.NUMBER_OF_WEEKS,
        toy.pi,
        toy.A,
        toy.means,
        toy.variances,
    )
    print(f"log-likelihood under the true parameters: {log_likelihood_true:.2f}")

    # 3. EM from the Householder start and the random starts
    fits, best_index = starting_values.fit_from_several_starts(
        X,
        weeks,
        toy.NUMBER_OF_STATES,
        toy.NUMBER_OF_WEEKS,
        number_of_random_starts,
        random_generator,
        max_iterations,
    )

    # 4. Put the states of every fit in the order MA, T, MU (closest true means), so A can be compared
    for fit in fits:
        fit["pi"], fit["A"], fit["means"], fit["variances"] = population_model.match_states(
            fit["pi"],
            fit["A"],
            fit["means"],
            fit["variances"],
            toy.means,
        )
    best = fits[best_index]

    # 5. Best fit: modelled p_t = pi A^t, and observed soft proportions n_t / (cells in week t) from its r
    proportions_best = population_model.get_proportions(best["pi"], best["A"], toy.NUMBER_OF_WEEKS)
    r_best, _ = population_model.e_step(
        X,
        weeks,
        toy.NUMBER_OF_WEEKS,
        best["pi"],
        best["A"],
        best["means"],
        best["variances"],
    )
    soft_proportions_best = population_model.soft_counts(r_best, weeks, toy.NUMBER_OF_WEEKS) / cells_per_week

    # 6. Print every start: does it reach the best A? (the anchoring check)
    print("start        iterations   l start -> l final      max|A - A_best|  max|A - A_true|")
    for k, fit in enumerate(fits):
        log_likelihoods = fit["log_likelihoods"]
        print(f"{k} {fit['start']:<11} {len(log_likelihoods) - 1:>10}   "
              f"{log_likelihoods[0]:>10.2f} -> {log_likelihoods[-1]:>9.2f}"
              f"{np.abs(fit['A'] - best['A']).max():>17.3f}{np.abs(fit['A'] - toy.A).max():>17.3f}"
              f"{'   <- best' if k == best_index else ''}")
    print("best fit A (rows and columns MA, T, MU):\n", best["A"].round(3))
    for j, state in enumerate(toy.STATES):
        print(f"row {state}: largest |A_fit - A_true| = {np.abs(best['A'][j] - toy.A[j]).max():.3f}")

    # 7. Save everything the plots and the report need
    path = save_results(f"population_recovery_{cells_per_week}", {
        "cells_per_week": cells_per_week, "number_of_random_starts": number_of_random_starts,
        "max_iterations": max_iterations, "seed": seed,
        "number_of_weeks": toy.NUMBER_OF_WEEKS, "states": toy.STATES,
        "cells": {"X": X, "weeks": weeks, "states": states},
        "true": {
            "pi": toy.pi,
            "A": toy.A,
            "means": toy.means,
            "variances": toy.variances,
            "proportions": population_model.get_proportions(toy.pi, toy.A, toy.NUMBER_OF_WEEKS),
            "log_likelihood": log_likelihood_true,
        },
        "fits": fits,
        "best_index": best_index,
        "best": {"proportions": proportions_best, "soft_proportions": soft_proportions_best},
    })
    print("saved", path)


if __name__ == "__main__":
    # run_baum_welch_restarts()
    run_population_recovery(cells_per_week=300)
    run_population_recovery(cells_per_week=3000)
