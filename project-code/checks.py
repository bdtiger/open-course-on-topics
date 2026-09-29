"""
Sanity checks for the algorithms in hmm.py, on a short fixed sequence.
Run after changing anything in hmm.py:  ../.venv/bin/python checks.py
"""
import numpy as np

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


if __name__ == "__main__":
    run_sanity_checks()
    print("all checks passed")
