"""
Sanity checks for the algorithms in hmm.py (on a short fixed sequence), population_model.py and
starting_values.py (with the toy constants of toy_parameters.py).
Run after changing anything in these files:  ../.venv/bin/python checks.py
"""
from itertools import permutations

import numpy as np
from scipy.stats import multivariate_normal

import population_model
import starting_values
import toy_parameters as toy   # toy.pi, toy.A, ...; hmm.py below has its own pi, A, STATES

from hmm import (STATES, SYMBOLS, pi, A, B, validate_hmm_parameters, sample_hmm_sequence,
                 forward_algorithm, backward_algorithm, posteriors, m_step)


def run_sanity_checks():
    """Forward, backward, posteriors and one M-step on a short fixed sequence, with assertions."""
    states, observations = sample_hmm_sequence(pi, A, B, 5, np.random.default_rng(0))
    observations = np.array([SYMBOLS.index(s) for s in ["B", "C", "C", "A", "A"]])  # Override with a fixed observation sequence for testing
    print("states:", states)
    print("observations:", observations)
    alpha, scale_factors, log_likelihood_forward = forward_algorithm(pi, A, B, observations)
    beta, log_likelihood_backward = backward_algorithm(pi, A, B, observations, scale_factors)
    gamma, xi = posteriors(pi, A, B, observations, alpha, beta, scale_factors)

    print("alpha:\n", alpha)
    print("beta:\n", beta)
    print("gamma:\n", gamma)
    print("xi:\n", xi)

    print("log-likelihood (forward):", log_likelihood_forward)
    print("log-likelihood (backward):", log_likelihood_backward)

    # Forward and backward checks: two ways, same number
    assert np.isclose(log_likelihood_forward, log_likelihood_backward)
    assert np.allclose(alpha.sum(axis=1), 1)
    assert np.allclose((alpha * beta).sum(axis=1), 1)

    # Posterior checks
    assert np.allclose(gamma.sum(axis=1), 1)
    assert np.allclose(xi.sum(axis=(1, 2)), 1)
    assert np.allclose(xi.sum(axis=2), gamma[:-1])  # summing xi over j gives gamma

    # Decoding check on a long sampled sequence (the fixed sequence above has no true states to compare with)
    long_states, long_observations = sample_hmm_sequence(pi, A, B, 2000, np.random.default_rng(0))
    long_alpha, long_scale_factors, _ = forward_algorithm(pi, A, B, long_observations)
    long_beta, _ = backward_algorithm(pi, A, B, long_observations, long_scale_factors)
    long_gamma, _ = posteriors(pi, A, B, long_observations, long_alpha, long_beta, long_scale_factors)
    accuracy = np.mean(long_gamma.argmax(axis=1) == long_states)
    print(f"argmax(gamma) matches the true state at {accuracy:.1%} of 2000 sampled time steps")
    assert accuracy > np.bincount(long_states).max() / len(long_states)  # better than always guessing the commonest state

    # M-step checks: one re-estimation from the true parameters gives valid distributions
    new_pi, new_A, new_B = m_step(observations, gamma, xi, len(SYMBOLS))
    validate_hmm_parameters(new_pi, new_A, new_B)


def run_population_checks():
    """Population model (population_model.py): proportions against hand-computed values."""
    proportions = population_model.get_proportions(toy.pi, toy.A, toy.NUMBER_OF_WEEKS)
    print("proportions (rows are p_0 .. p_3):\n", proportions)

    # Hand values from planning-doc/hand-check/answers.md, Part 1 (exact)
    proportions_hand = np.array([
        [0.900000, 0.050000, 0.050000],
        [0.726000, 0.166500, 0.107500],
        [0.599600, 0.212025, 0.188375],
        [0.504650, 0.22280625, 0.27254375],
    ])
    assert proportions.shape == (toy.NUMBER_OF_WEEKS, len(toy.STATES))
    assert np.allclose(proportions, proportions_hand, atol=1e-10)
    assert np.allclose(proportions.sum(axis=1), 1)

    # Matches the closed form p_t = pi A^t
    for t in range(toy.NUMBER_OF_WEEKS):
        assert np.allclose(proportions[t], toy.pi @ np.linalg.matrix_power(toy.A, t))

    # Simulator: with many cells, the sample matches the parameters it was drawn from
    cells_per_week = 3000
    X, weeks, states = population_model.sample_cells(
        toy.pi,
        toy.A,
        toy.means,
        toy.variances,
        cells_per_week,
        toy.NUMBER_OF_WEEKS,
        np.random.default_rng(0),
    )
    n = toy.NUMBER_OF_WEEKS * cells_per_week
    assert X.shape == (n, toy.NUMBER_OF_GENES)
    assert weeks.shape == (n,) and states.shape == (n,)
    assert np.all(np.bincount(weeks) == cells_per_week)

    # Fraction of each hidden state per week is close to p_t
    observed_proportions = np.array([np.bincount(states[weeks == t], minlength=len(toy.STATES))
                                     / cells_per_week for t in range(toy.NUMBER_OF_WEEKS)])
    print("observed proportions (3000 cells per week):\n", observed_proportions)
    assert np.allclose(observed_proportions, proportions, atol=0.02)

    # Per-state sample mean and variance are close to mu_j and 1 (all weeks pooled)
    for j in range(len(toy.STATES)):
        X_j = X[states == j]
        print(f"{toy.STATES[j]}: {len(X_j)} cells, mean {X_j.mean(axis=0).round(3)}, "
              f"variance {X_j.var(axis=0).round(3)}")
        assert np.allclose(X_j.mean(axis=0), toy.means[j], atol=0.1)
        assert np.allclose(X_j.var(axis=0), toy.variances[j], atol=0.1)

    # Same seed gives the same cells
    X_again, _, states_again = population_model.sample_cells(
        toy.pi,
        toy.A,
        toy.means,
        toy.variances,
        cells_per_week,
        toy.NUMBER_OF_WEEKS,
        np.random.default_rng(0),
    )
    assert np.array_equal(X, X_again) and np.array_equal(states, states_again)

    # Gaussian log-densities: the 4 hand-check cells (planning-doc/hand-check/answers.md, Part 2)
    X_hand = np.array([[4.0, 2.0], [3.0, 2.0], [2.0, 3.0], [1.0, 5.0]])
    squared_distances_hand = np.array([   # columns: to MA, T, MU
        [1.0, 2.5, 13.0],
        [2.0, 0.5, 8.0],
        [8.0, 0.5, 2.0],
        [25.0, 8.5, 1.0],
    ])
    log_densities_hand = -np.log(2 * np.pi) - 0.5 * squared_distances_hand  # -1.8379 - 1/2 * distance
    log_densities = population_model.log_gaussian_densities(X_hand, toy.means, toy.variances)
    print("log-densities of the 4 hand-check cells (columns MA, T, MU):\n", log_densities.round(4))
    assert log_densities.shape == (4, len(toy.STATES))
    assert np.allclose(log_densities, log_densities_hand)
    assert np.isclose(log_densities[0, 0], -2.3379, atol=1e-4)  # spot check against the printed table

    # Same as scipy on random inputs, with unequal variances and d = 5
    random_generator = np.random.default_rng(1)
    X_random = random_generator.normal(size=(10, 5))
    means_random = random_generator.normal(size=(3, 5))
    variances_random = random_generator.uniform(0.2, 3.0, size=(3, 5))
    log_densities_random = population_model.log_gaussian_densities(X_random, means_random, variances_random)
    for j in range(3):
        log_densities_scipy = multivariate_normal.logpdf(
            X_random,
            mean=means_random[j],
            cov=np.diag(variances_random[j]),
        )
        assert np.allclose(log_densities_random[:, j], log_densities_scipy)

    # A cell far from every mean gives a very negative but finite number, never -inf
    log_densities_far = population_model.log_gaussian_densities(
        np.array([[1000.0, -1000.0]]),
        toy.means,
        toy.variances,
    )
    print("log-densities of a far cell (1000, -1000):", log_densities_far)
    assert np.all(np.isfinite(log_densities_far))

    # E-step: the 4 hand-check cells, one per week (planning-doc/hand-check/answers.md, Part 3)
    weeks_hand = np.array([0, 1, 2, 3])
    r, log_likelihood = population_model.e_step(
        X_hand,
        weeks_hand,
        toy.NUMBER_OF_WEEKS,
        toy.pi,
        toy.A,
        toy.means,
        toy.variances,
    )
    r_hand = np.array([   # columns: MA, T, MU
        [0.9743, 0.0256, 0.0001],
        [0.6698, 0.3252, 0.0049],
        [0.0448, 0.6729, 0.2824],
        [0.0000, 0.0189, 0.9811],
    ])
    print("responsibilities of the 4 hand-check cells (columns MA, T, MU):\n", r.round(4))
    print("log-likelihood of the 4 hand-check cells:", round(log_likelihood, 4))
    assert r.shape == (4, len(toy.STATES))
    assert np.allclose(r, r_hand, atol=1e-4)
    assert np.allclose(r.sum(axis=1), 1)
    assert np.isclose(log_likelihood, -12.0361, atol=1e-4)

    # Log-likelihood equals the direct product of sums, computed without logs (5 sampled cells)
    X_five, weeks_five, _ = population_model.sample_cells(
        toy.pi,
        toy.A,
        toy.means,
        toy.variances,
        2,
        toy.NUMBER_OF_WEEKS,
        np.random.default_rng(2),
    )
    X_five, weeks_five = X_five[:5], weeks_five[:5]
    _, log_likelihood_five = population_model.e_step(
        X_five,
        weeks_five,
        toy.NUMBER_OF_WEEKS,
        toy.pi,
        toy.A,
        toy.means,
        toy.variances,
    )
    proportions_five = proportions[weeks_five]                                  # prior of each cell, (5, N)
    densities_five = np.exp(population_model.log_gaussian_densities(X_five, toy.means, toy.variances))
    likelihood_direct = np.prod(np.sum(proportions_five * densities_five, axis=1))
    assert np.isclose(log_likelihood_five, np.log(likelihood_direct))

    # Gaussian update: the 4 hand-check cells and their r (planning-doc/hand-check/answers.md, Part 4)
    soft_cell_counts_hand = np.array([1.6889, 1.0425, 1.2686])   # 4a, sum_c r_cj per state
    means_hand = np.array([[3.5504, 2.0265], [2.3429, 2.6997], [1.2307, 4.5428]])   # 4b
    variances_hand = np.array([[0.3005, 0.0259], [0.3106, 0.3187], [0.1859, 0.7174]])   # 4c
    new_means, new_variances = population_model.m_step(X_hand, r)
    print("new means after one Gaussian update (rows MA, T, MU):\n", new_means.round(4))
    print("new variances after one Gaussian update (rows MA, T, MU):\n", new_variances.round(4))
    assert np.allclose(r.sum(axis=0), soft_cell_counts_hand, atol=1e-4)
    assert np.allclose(new_means, means_hand, atol=1e-4)
    assert np.allclose(new_variances, variances_hand, atol=1e-4)

    # The update raises l: -12.0361 before, -5.0286 after (answers.md, Part 5)
    _, log_likelihood_after = population_model.e_step(
        X_hand,
        weeks_hand,
        toy.NUMBER_OF_WEEKS,
        toy.pi,
        toy.A,
        new_means,
        new_variances,
    )
    print("log-likelihood of the 4 hand-check cells after the Gaussian update:", round(log_likelihood_after, 4))
    assert np.isclose(log_likelihood_after, -5.0286, atol=1e-4)
    assert log_likelihood_after > log_likelihood

    # One-hot r from the true states (each cell fully in its own state) gives the per-state
    # sample mean and variance of the 3000-cells-per-week sample
    r_one_hot = np.eye(len(toy.STATES))[states]   # row c is 1 in column states[c], 0 elsewhere
    one_hot_means, one_hot_variances = population_model.m_step(X, r_one_hot)
    for j in range(len(toy.STATES)):
        assert np.allclose(one_hot_means[j], X[states == j].mean(axis=0))
        assert np.allclose(one_hot_variances[j], X[states == j].var(axis=0))

    # Variance floor: a state that owns only one cell has variance 0, raised to the floor
    r_one_cell = np.eye(len(toy.STATES))[[0, 0, 1, 2]]   # T owns only cell 3, MU only cell 4
    _, floored_variances = population_model.m_step(X_hand, r_one_cell, variance_floor=1e-3)
    assert np.all(floored_variances >= 1e-3)
    assert np.allclose(floored_variances[1:], 1e-3)

    # Householder start with the true p0, p1 (planning-doc/hand-check/answers.md, Part 7f)
    A_start = starting_values.householder_start(proportions[0], proportions[1], floor=0.01)
    A_start_hand = np.array([   # rows and columns: MA, T, MU
        [0.0077, 0.6644, 0.3279],
        [0.6644, 0.3279, 0.0077],
        [0.3279, 0.0077, 0.6644],
    ])
    print("Householder start A from the true p0, p1:\n", A_start.round(4))
    assert np.allclose(A_start, A_start_hand, atol=1e-4)
    assert np.allclose(A_start.sum(axis=1), 1)
    assert np.all(A_start > 0)
    population_model.validate_population_parameters(proportions[0], A_start, toy.means, toy.variances)

    # Random starts: valid, no zeros, same seed gives the same start
    for seed in range(5):
        pi_random, A_random = starting_values.random_start(len(toy.STATES), np.random.default_rng(seed))
        population_model.validate_population_parameters(pi_random, A_random, toy.means, toy.variances)
        assert np.all(pi_random > 0) and np.all(A_random > 0)
    pi_again, A_again = starting_values.random_start(len(toy.STATES), np.random.default_rng(4))
    assert np.array_equal(pi_random, pi_again) and np.array_equal(A_random, A_again)

    # k-means start on the 3000-cells-per-week sample: cluster fractions per week are probability
    # distributions, and the full start is valid. Not checked: that each centre is near a true mean.
    # MA is about 2/3 of all cells, so k-means splits MA in two and finds no T centre; EM must fix that.
    kmeans_means, kmeans_variances, cluster_proportions = starting_values.initial_gaussians(
        X,
        weeks,
        len(toy.STATES),
        toy.NUMBER_OF_WEEKS,
        np.random.default_rng(0),
    )
    print("k-means centres (arbitrary order):\n", kmeans_means.round(3))
    print("cluster fractions per week (rows week 0..3):\n", cluster_proportions.round(3))
    assert kmeans_means.shape == toy.means.shape
    assert np.allclose(cluster_proportions.sum(axis=1), 1)
    A_kmeans_start = starting_values.householder_start(cluster_proportions[0], cluster_proportions[1])
    population_model.validate_population_parameters(
        cluster_proportions[0],
        A_kmeans_start,
        kmeans_means,
        kmeans_variances,
    )

    # Soft counts: one hand-check cell per week, so n_t = r of that cell; rows of a sample sum to its cells per week
    counts_hand = population_model.soft_counts(r, weeks_hand, toy.NUMBER_OF_WEEKS)
    assert np.allclose(counts_hand, r)
    assert np.allclose(
        population_model.soft_counts(r_one_hot, weeks, toy.NUMBER_OF_WEEKS).sum(axis=1),
        cells_per_week,
    )

    # Q of the hand-check cells with the true pi, A (answers.md, Part 6): -3.8299, below Q_free = -1.6367
    Q_true = population_model.expected_log_weights(toy.pi, toy.A, counts_hand)
    print("Q of the 4 hand-check cells with the true pi, A:", round(Q_true, 4))
    assert np.isclose(Q_true, -3.8299, atol=1e-4)
    assert Q_true <= -1.6367

    # Exact proportions n = 10^5 * p_true: the best of 5 random starts returns the true pi, A.
    # Not every start: Q has local maxima (random start 0 ends in one), so starts are compared by Q.
    counts_exact = 1e5 * proportions
    best_Q = -np.inf
    for seed in range(5):
        pi_random, A_random = starting_values.random_start(len(toy.STATES), np.random.default_rng(seed))
        Q_before = population_model.expected_log_weights(pi_random, A_random, counts_exact)
        pi_fit, A_fit = population_model.update_pi_A(counts_exact, pi_random, A_random, max_iterations=5000)
        Q_after = population_model.expected_log_weights(pi_fit, A_fit, counts_exact)
        assert Q_after >= Q_before   # the update never lowers Q
        if Q_after > best_Q:
            best_Q, best_pi, best_A = Q_after, pi_fit, A_fit
    print("pi, A from exact proportions (best of 5 random starts):\n", best_pi.round(4), "\n", best_A.round(4))
    assert np.allclose(best_pi, toy.pi, atol=1e-3)
    assert np.allclose(best_A, toy.A, atol=1e-3)

    # Matching states: every shuffled copy of the true parameters is put back in the order MA, T, MU exactly.
    # A shuffle only renames the states, so l of the hand-check cells is unchanged.
    for shuffle in permutations(range(len(toy.STATES))):
        shuffle = list(shuffle)
        pi_shuffled = toy.pi[shuffle]
        A_shuffled = toy.A[shuffle][:, shuffle]
        means_shuffled = toy.means[shuffle]
        variances_shuffled = toy.variances[shuffle]

        _, log_likelihood_shuffled = population_model.e_step(
            X_hand,
            weeks_hand,
            toy.NUMBER_OF_WEEKS,
            pi_shuffled,
            A_shuffled,
            means_shuffled,
            variances_shuffled,
        )
        assert np.isclose(log_likelihood_shuffled, log_likelihood)

        pi_matched, A_matched, means_matched, variances_matched = population_model.match_states(
            pi_shuffled,
            A_shuffled,
            means_shuffled,
            variances_shuffled,
            toy.means,
        )
        assert np.array_equal(pi_matched, toy.pi)
        assert np.array_equal(A_matched, toy.A)
        assert np.array_equal(means_matched, toy.means)
        assert np.array_equal(variances_matched, toy.variances)
    print("match_states puts all 6 shuffled copies of the true parameters back in the order", toy.STATES)

    # Several starts on a small sample (100 cells per week): the Householder fit first, then the random fits;
    # every fit is valid with no exact 0 in pi or A, and best_index points at the highest final l
    X_small, weeks_small, _ = population_model.sample_cells(
        toy.pi,
        toy.A,
        toy.means,
        toy.variances,
        100,
        toy.NUMBER_OF_WEEKS,
        np.random.default_rng(3),
    )
    fits, best_index = starting_values.fit_from_several_starts(
        X_small,
        weeks_small,
        len(toy.STATES),
        toy.NUMBER_OF_WEEKS,
        2,
        np.random.default_rng(3),
    )
    assert [fit["start"] for fit in fits] == ["householder", "random", "random"]
    for fit in fits:
        population_model.validate_population_parameters(fit["pi"], fit["A"], fit["means"], fit["variances"])
        assert np.all(fit["pi"] > 0) and np.all(fit["A"] > 0)
    final_log_likelihoods = [fit["log_likelihoods"][-1] for fit in fits]
    assert best_index == np.argmax(final_log_likelihoods)
    print("fit_from_several_starts, final l of each start:", np.round(final_log_likelihoods, 2))


if __name__ == "__main__":
    run_sanity_checks()
    run_population_checks()
    print("all checks passed")
