
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
    Check that pi, A and B agree on the number of states and that every probability row sums to 1.

    Parameters
    ----------
    pi: (N,) initial state probabilities
    A:  (N, N) transition probabilities, rows sum to 1
    B:  (N, M) emission probabilities, rows sum to 1

    Returns
    -------
    None; raises AssertionError if a shape or a row sum is wrong
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
    Draw one index from a probability row (a dart thrown at the cumulative sums).

    Parameters
    ----------
    probability_row: (K,) probabilities summing to 1
    random_generator: numpy Generator, e.g. np.random.default_rng(0)

    Returns
    -------
    int in 0..K-1, the drawn index
    """
    cumulative_sum = np.cumsum(probability_row) # Return the array of cumulative sums of the probability row
    random_value = random_generator.random()
    return np.searchsorted(cumulative_sum, random_value)


def sample_hmm_sequence(pi, A, B, T, random_generator):
    """
    Sample a sequence of hidden states and observations from the HMM (pi, A, B).

    Parameters
    ----------
    pi: (N,) initial state probabilities
    A:  (N, N) transition probabilities
    B:  (N, M) emission probabilities
    T:  int, length of the sequence
    random_generator: numpy Generator; same seed gives the same sequence

    Returns
    -------
    states: (T,) int in 0..N-1, the hidden state at each time step
    observations: (T,) int in 0..M-1, the emitted symbol at each time step
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


def random_parameters(N, M, random_generator):
    """
    Draw a random starting point for Baum-Welch: every row a random probability distribution (uniform Dirichlet).

    Parameters
    ----------
    N: int, number of hidden states
    M: int, number of observation symbols
    random_generator: numpy Generator; different seed gives a different start

    Returns
    -------
    pi: (N,) random initial state probabilities
    A:  (N, N) random transition probabilities, rows sum to 1
    B:  (N, M) random emission probabilities, rows sum to 1
    """
    pi = random_generator.dirichlet(np.ones(N))
    A = random_generator.dirichlet(np.ones(N), size=N)
    B = random_generator.dirichlet(np.ones(M), size=N)
    return pi, A, B


def forward_algorithm(pi, A, B, observations):
    """
    Forward pass with per-step scaling.
    alpha[t, i] is the probability of being in state i at time t, given the observations up to time t.
    Every row of alpha is divided by its sum so the numbers do not shrink to zero on long sequences;
    the sum divided out at time t is saved in scale_factors[t], and log P(O) = sum_t log scale_factors[t].

    Parameters
    ----------
    pi: (N,) initial state probabilities
    A:  (N, N) transition probabilities
    B:  (N, M) emission probabilities
    observations: (T,) int in 0..M-1

    Returns
    -------
    alpha: (T, N) scaled forward probabilities, every row sums to 1
    scale_factors: (T,) the row sum divided out at each time step
    log_likelihood: float, log P(O | lambda)
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
    Backward pass with the scaling from the forward pass.
    beta[t, i] is the probability of the observations after time t, given state i at time t.
    Row t of beta is divided by scale_factors[t+1], so alpha[t] * beta[t] sums to 1 at every t.

    Parameters
    ----------
    pi: (N,) initial state probabilities
    A:  (N, N) transition probabilities
    B:  (N, M) emission probabilities
    observations: (T,) int in 0..M-1
    scale_factors: (T,) from forward_algorithm on the same observations

    Returns
    -------
    beta: (T, N) scaled backward probabilities
    log_likelihood: float, must equal the forward value (the scaling check)
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
    Posteriors gamma and xi from alpha and beta.
    gamma[t, i] is the probability of being in state i at time t, given the whole sequence.
    xi[t, i, j] is the probability of moving from state i at time t to state j at time t+1, given the whole sequence.

    Parameters
    ----------
    pi: (N,) unused, kept so all E-step functions take the same arguments
    A:  (N, N) transition probabilities
    B:  (N, M) emission probabilities
    observations: (T,) int in 0..M-1
    alpha: (T, N) from forward_algorithm
    beta:  (T, N) from backward_algorithm
    scale_factors: (T,) from forward_algorithm

    Returns
    -------
    gamma: (T, N) state posteriors, every row sums to 1
    xi: (T-1, N, N) transition posteriors, every xi[t] sums to 1
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
    E-step of Baum-Welch: forward pass, backward pass, then the posteriors gamma and xi.

    Parameters
    ----------
    pi: (N,) current initial state probabilities
    A:  (N, N) current transition probabilities
    B:  (N, M) current emission probabilities
    observations: (T,) int in 0..M-1

    Returns
    -------
    gamma: (T, N) state posteriors
    xi: (T-1, N, N) transition posteriors
    log_likelihood: float, log P(O | lambda) under the current parameters
    """
    alpha, scale_factors, log_likelihood = forward_algorithm(pi, A, B, observations)
    beta, _ = backward_algorithm(pi, A, B, observations, scale_factors)
    gamma, xi = posteriors(pi, A, B, observations, alpha, beta, scale_factors)
    return gamma, xi, log_likelihood


def m_step(observations, gamma, xi, M):
    """
    M-step of Baum-Welch. Turn the posteriors gamma and xi into new parameters.
    Every new parameter is a normalised expected count:
      pi_i    <- gamma_1(i)
      A[i, j] <- sum_{t=1}^{T-1} xi_t(i, j)  /  sum_{t=1}^{T-1} gamma_t(i)
      B[j, k] <- sum_{t : o_t = k} gamma_t(j) /  sum_t gamma_t(j)

    Parameters
    ----------
    observations: (T,) int in 0..M-1
    gamma: (T, N) state posteriors from the E-step
    xi:    (T-1, N, N) transition posteriors from the E-step
    M: int, number of observation symbols

    Returns
    -------
    new_pi: (N,) re-estimated initial state probabilities
    new_A:  (N, N) re-estimated transition probabilities, rows sum to 1
    new_B:  (N, M) re-estimated emission probabilities, rows sum to 1
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
    until the log-likelihood improves by less than tolerance.

    Parameters
    ----------
    pi: (N,) starting initial state probabilities (e.g. from random_parameters)
    A:  (N, N) starting transition probabilities
    B:  (N, M) starting emission probabilities
    observations: (T,) int in 0..M-1
    max_iterations: int, upper limit on EM iterations (default 100)
    tolerance: float, stop when the improvement is smaller than this (default 1e-6)

    Returns
    -------
    pi: (N,) fitted initial state probabilities
    A:  (N, N) fitted transition probabilities
    B:  (N, M) fitted emission probabilities
    log_likelihoods: list of float, one per iteration, never decreasing
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
