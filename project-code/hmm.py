
import numpy as np

# Shape conventions
#   N = number of hidden states   (A is N x N)
#   M = number of symbols         (B is N x M)
#   T = sequence length           (observations has length T)
#   pi: (N,)   A: (N, N)   B: (N, M)
#   alpha, beta, gamma: (T, N)    xi: (T-1, N, N)


STATES = ["S1", "S2", "S3"]
SYMBOLS = ["A", "B", "C", "D"]

pi = np.array([0.6, 0.3, 0.1])  # Initial state probabilities

A = np.array([
    [0.7, 0.2, 0.1],  # Transition probabilities from S1
    [0.1, 0.3, 0.6],  # Transition probabilities from S2
    [0.2, 0.3, 0.5]   # Transition probabilities from S3
])

B = np.array([
    [0.7, 0.1, 0.1, 0.1],  # Emission probabilities for S1
    [0.1, 0.7, 0.1, 0.1],  # Emission probabilities for S2
    [0.1, 0.1, 0.4, 0.4]   # Emission probabilities for S3
])


def validate_hmm_parameters(pi, A, B):
    """
    Check that pi, A and B agree on the number of states and that every
    probability row sums to 1. N is taken from pi (one entry per state).
    """
    N = len(pi)
    M = len(SYMBOLS)
    assert pi.ndim == 1 and N > 0, "pi must be a non-empty 1-D array"
    assert A.shape == (N, N), "A must be N x N"
    assert B.shape == (N, M), "B must be N x M (one row per state, one column per symbol)"
    assert np.isclose(np.sum(pi), 1), "Initial state probabilities must sum to 1."
    assert np.allclose(A.sum(axis=1), 1), "Transition probabilities must sum to 1 for each state."
    assert np.allclose(B.sum(axis=1), 1), "Emission probabilities must sum to 1 for each state."


validate_hmm_parameters(pi, A, B)


def pick_from_row(probability_row, random_generator):
    """
    Return the zone dart landed in based on the given probability row and a random number generator.
    """
    cumulative_sum = np.cumsum(probability_row) # Return the array of cumulative sums of the probability row
    random_value = random_generator.random()
    return np.searchsorted(cumulative_sum, random_value)


def sample_hmm_sequence(pi, A, B, T, random_generator):
    """
    Sample a sequence of states and observations from the HMM defined by 
    the initial state probabilities (pi), transition probabilities (A), and emission probabilities (B).
    """
    states = np.zeros(T, dtype=int)
    observations = np.zeros(T, dtype=int)

    # Sample the initial state
    first_state = pick_from_row(pi, random_generator)
    states[0] = first_state
    observations[0] = pick_from_row(B[first_state], random_generator)

    # Sample the rest of the states and observations
    for t in range(1, T):
        current_state = states[t - 1]
        next_state = pick_from_row(A[current_state], random_generator)
        states[t] = next_state
        observations[t] = pick_from_row(B[next_state], random_generator)

    return states, observations

def forward_algorithm(pi, A, B, observations):
    """
    Compute the forward probabilities (alpha) for a sequence of observations.
    alpha[t, i] is the probability of being in state i at time t, given the observations up to time t.

    Every row of alpha is divided by its sum so the numbers do not shrink to zero on long sequences.
    The sum that was divided out at time t is saved in scale_factors[t].
    The likelihood is the product of all scale factors, so the log-likelihood is the sum of their logs.

    Returns alpha (T x N, each row sums to 1), scale_factors (length T) and log_likelihood.
    """
    T = len(observations)
    N = len(pi)
    alpha = np.zeros((T, N))
    scale_factors = np.zeros(T)

    # Initialise: alpha_1(i) = pi_i * B[i, o_1], then divide the row by its sum
    alpha_row = pi * B[:, observations[0]]
    scale_factors[0] = alpha_row.sum()
    alpha[0] = alpha_row / scale_factors[0]

    # Recurse: alpha_t(j) = (sum_i alpha_{t-1}(i) * A[i, j]) * B[j, o_t], then divide the row by its sum
    for t in range(1, T):
        alpha_row = np.zeros(N)
        for j in range(N):
            transition_sum = alpha[t-1] @ A[:, j]  # dot product = sum_i alpha[t-1, i] * A[i, j]
            alpha_row[j] = transition_sum * B[j, observations[t]]
        scale_factors[t] = alpha_row.sum()
        alpha[t] = alpha_row / scale_factors[t]

    # Terminate: log P(O) = sum_t log c_t
    log_likelihood = np.sum(np.log(scale_factors))
    return alpha, scale_factors, log_likelihood


def backward_algorithm(pi, A, B, observations, scale_factors):
    """
    Compute the backward probabilities (beta) for a sequence of observations.
    beta[t, i] is the probability of the observations after time t, given state i at time t.

    Row t of beta is divided by scale_factors[t+1], the same number the forward pass divided out at time t+1.
    This keeps alpha and beta on the same scale, so alpha[t] * beta[t] sums to 1 at every t.

    Returns beta (T x N) and log_likelihood. The log-likelihood must match the one from the forward pass.
    """
    T = len(observations)
    N = len(pi)
    beta = np.zeros((T, N))
    beta[T-1] = 1  # Initialise: beta_T(i) = 1 for all i

    # Recurse backwards: beta_t(i) = sum_j A[i, j] * B[j, o_{t+1}] * beta_{t+1}(j), then divide by c_{t+1}
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i, :] * B[:, observations[t+1]] * beta[t+1])
        beta[t] /= scale_factors[t+1]

    # Terminate: P(O) = sum_i pi_i * B[i, o_1] * beta_1(i). beta[0] was scaled by c_2 .. c_T, so add those logs back
    first_step = np.sum(pi * B[:, observations[0]] * beta[0])
    log_likelihood = np.log(first_step) + np.sum(np.log(scale_factors[1:]))
    return beta, log_likelihood


def posteriors(pi, A, B, observations, alpha, beta, scale_factors):
    """
    Compute the posteriors gamma and xi from alpha and beta. This is the E-step.
    gamma[t, i] is the probability of being in state i at time t, given the whole sequence.
    xi[t, i, j] is the probability of moving from state i at time t to state j at time t+1, given the whole sequence.

    alpha and beta share the same scaling, so gamma = alpha * beta already sums to 1 in every row.
    xi is divided by scale_factors[t+1] so every t-slice sums to 1.

    Returns gamma (T x N) and xi ((T-1) x N x N).
    """
    T = len(observations)
    N = len(pi)

    gamma = alpha * beta

    xi = np.zeros((T-1, N, N))
    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, observations[t+1]] * beta[t+1, j]
        xi[t] /= scale_factors[t+1]

    return gamma, xi


def e_step(pi, A, B, observations):
    """
    E-step of Baum-Welch: forward pass, backward pass, then the posteriors gamma and xi
    under the current parameters pi, A and B.

    Returns gamma (T x N), xi ((T-1) x N x N) and the log-likelihood log P(O | lambda).
    """
    alpha, scale_factors, log_likelihood = forward_algorithm(pi, A, B, observations)
    beta, _ = backward_algorithm(pi, A, B, observations, scale_factors)
    gamma, xi = posteriors(pi, A, B, observations, alpha, beta, scale_factors)
    return gamma, xi, log_likelihood


def m_step(observations, gamma, xi, M):
    """
    Baum-Welch re-estimation. Turn the posteriors gamma and xi into new parameters.
    Every new parameter is a normalised expected count:
      pi_i    <- gamma_1(i)
      A[i, j] <- sum_{t=1}^{T-1} xi_t(i, j)  /  sum_{t=1}^{T-1} gamma_t(i)
      B[j, k] <- sum_{t : o_t = k} gamma_t(j) /  sum_t gamma_t(j)

    Returns new pi (N,), A (N x N) and B (N x M).
    """
    N = gamma.shape[1]

    # pi: posterior of the first state
    new_pi = gamma[0].copy()

    # A: expected transitions i -> j over expected visits to i (last time step excluded, no transition leaves it)
    expected_transitions = xi.sum(axis=0)              # (N, N)
    expected_visits = gamma[:-1].sum(axis=0)           # (N,)
    new_A = expected_transitions / expected_visits[:, None]

    # B: expected emissions of symbol k from state j over expected visits to j
    new_B = np.zeros((N, M))
    for k in range(M):
        new_B[:, k] = gamma[observations == k].sum(axis=0)
    new_B /= gamma.sum(axis=0)[:, None]

    return new_pi, new_A, new_B


def baum_welch(pi, A, B, observations, max_iterations=100, tolerance=1e-6):
    """
    Fit pi, A and B to the observations by repeating e_step and m_step
    until the log-likelihood stops increasing by more than tolerance.

    Returns the fitted pi, A, B and the list of log-likelihoods, one per iteration.
    """
    M = B.shape[1]
    log_likelihoods = []
    for _ in range(max_iterations):
        gamma, xi, log_likelihood = e_step(pi, A, B, observations)
        log_likelihoods.append(log_likelihood)

        pi, A, B = m_step(observations, gamma, xi, M)

        if len(log_likelihoods) > 1 and log_likelihoods[-1] - log_likelihoods[-2] < tolerance:
            break

    return pi, A, B, log_likelihoods


if __name__ == "__main__":
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

    decoded_states = gamma.argmax(axis=1)
    accuracy = np.mean(decoded_states == states)
    print(f"log-likelihood = {log_likelihood_forward:.3f}")
    print(f"argmax(gamma) matches the true state at {accuracy:.1%} of time steps")

    # M-step checks: one re-estimation from the true parameters gives valid distributions
    new_pi, new_A, new_B = m_step(observations, gamma, xi, len(SYMBOLS))
    validate_hmm_parameters(new_pi, new_A, new_B)

    # Baum-Welch on a long sequence: start from a random guess, the log-likelihood must never decrease
    _, long_observations = sample_hmm_sequence(pi, A, B, 2000, np.random.default_rng(0))
    rng = np.random.default_rng(1)
    pi_guess = rng.dirichlet(np.ones(len(STATES)))
    A_guess = rng.dirichlet(np.ones(len(STATES)), size=len(STATES))
    B_guess = rng.dirichlet(np.ones(len(SYMBOLS)), size=len(STATES))
    pi_fit, A_fit, B_fit, log_likelihoods = baum_welch(pi_guess, A_guess, B_guess, long_observations)
    validate_hmm_parameters(pi_fit, A_fit, B_fit)
    assert np.all(np.diff(log_likelihoods) >= -1e-8), "log-likelihood decreased during Baum-Welch"

    _, _, log_likelihood_true = forward_algorithm(pi, A, B, long_observations)
    print(f"Baum-Welch on {len(long_observations)} observations: {len(log_likelihoods)} iterations, "
          f"log-likelihood {log_likelihoods[0]:.1f} -> {log_likelihoods[-1]:.1f} "
          f"(true parameters: {log_likelihood_true:.1f})")
    # Note: the fitted states may come out in a different order than S1, S2, S3 (label switching)
    np.set_printoptions(precision=3, suppress=True)
    print("fitted A =\n", A_fit)
    print("fitted B =\n", B_fit)
