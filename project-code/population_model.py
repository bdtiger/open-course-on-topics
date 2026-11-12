"""
Population model and its EM algorithm. Algorithms only: no data, no constants,
no prints. The toy parameters are in toy_parameters.py, the starting values in starting_values.py.
"""
from itertools import permutations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

# Shape conventions
#   N = number of hidden states (A is N x N)
#   d = number of genes      (each cell is a vector of length d)
#   W = number of weeks      (4: day 0, 7, 14, 21)
#   pi: (N,)   A: (N, N)   proportions: (W, N), row t is p_t
#   means: (N, d)
#   variances: (N, d), the diagonal of U_j


def validate_population_parameters(pi, A, means, variances):
    """
    Check that pi, A, means and variances agree on the number of hidden states, that pi and every row
    of A are probability distributions, and that every variance is positive.

    Parameters
    ----------
    pi:        (N,) initial hidden state proportions
    A:         (N, N) one-week transition probabilities
    means:     (N, d) mean expression of each hidden state
    variances: (N, d) variance of each gene in each hidden state (diagonal of U_j)

    Returns
    -------
    None; raises AssertionError if a shape, a sign or a row sum is wrong
    """
    N = len(pi)
    assert pi.ndim == 1 and N > 0, "pi must be a non-empty 1-D array"
    assert A.shape == (N, N), "A must be N x N"
    assert means.ndim == 2 and means.shape[0] == N, "means must be N x d (one row per state)"
    assert variances.shape == means.shape, "variances must have the same shape as means"
    assert np.all(pi >= 0), "Initial state proportions must be non-negative."
    assert np.all(A >= 0), "Transition probabilities must be non-negative."
    assert np.isclose(np.sum(pi), 1), "Initial state proportions must sum to 1."
    assert np.allclose(A.sum(axis=1), 1), "Transition probabilities must sum to 1 for each state."
    assert np.all(variances > 0), "Variances must be positive."


def get_proportions(pi, A, number_of_weeks):
    """
    State proportions of the population in each week, p_t = pi A^t.

    Parameters
    ----------
    pi:      (N,) initial state proportions (week 0)
    A:       (N, N) one-week transition probabilities
    number_of_weeks: W, number of weeks including week 0

    Returns
    -------
    proportions: (W, N), row t is p_t; each row sums to 1
    """
    proportions = np.zeros((number_of_weeks, len(pi)))
    proportions[0] = pi
    for t in range(1, number_of_weeks):
        proportions[t] = proportions[t - 1] @ A  # Row vector times A: p_t = p_{t-1} A
    return proportions


def sample_cells(pi, A, means, variances, cells_per_week, number_of_weeks, random_generator):
    """
    Simulate measured cells: in each week, draw each cell's hidden state from p_t, then its
    expression vector from that state's Gaussian. Every cell is measured once; no chain per cell.

    Parameters
    ----------
    pi:             (N,) initial hidden state proportions (week 0)
    A:              (N, N) one-week transition probabilities
    means:          (N, d) mean expression of each hidden state
    variances:      (N, d) variance of each gene in each hidden state (diagonal of U_j)
    cells_per_week: int, number of cells measured in each week
    number_of_weeks: W, number of weeks including week 0
    random_generator: numpy Generator; same seed gives the same cells

    Returns
    -------
    X:      (n, d) expression of each cell, n = W * cells_per_week
    weeks:  (n,) int in 0..W-1, the week each cell was measured in
    states: (n,) int in 0..N-1, the true hidden state of each cell (answer key, not seen by the fit)
    """
    number_of_states, number_of_genes = means.shape
    proportions = get_proportions(pi, A, number_of_weeks)

    weeks = np.repeat(np.arange(number_of_weeks), cells_per_week)
    states = np.zeros(len(weeks), dtype=int)
    for t in range(number_of_weeks):
        in_week = weeks == t
        states[in_week] = random_generator.choice(number_of_states, size=cells_per_week, p=proportions[t])

    # x = mu_j + sigma_j * (standard normal noise), gene by gene (U_j is diagonal)
    noise = random_generator.standard_normal((len(weeks), number_of_genes))
    X = means[states] + np.sqrt(variances[states]) * noise
    return X, weeks, states


def log_gaussian_densities(X, means, variances):
    """
    Log-density of every cell under every hidden state's Gaussian (diagonal covariance U_j):

        log N(x_c; mu_j, U_j) = sum_g [ -1/2 log(2 pi sigma^2_jg) - (x_cg - mu_jg)^2 / (2 sigma^2_jg) ]

    One term per gene (U_j is diagonal, so the genes are independent given the state), then summed.
    Computed directly in log form; the density itself would round to 0 for many genes.

    Parameters
    ----------
    X:         (n, d) expression of each cell
    means:     (N, d) mean expression of each hidden state
    variances: (N, d) variance of each gene in each hidden state (diagonal of U_j)

    Returns
    -------
    log_densities: (n, N), entry [c, j] is log N(x_c; mu_j, U_j)
    """
    # Line up every cell with every state: (n, 1, d) against (1, N, d) -> (n, N, d)
    differences = X[:, np.newaxis, :] - means[np.newaxis, :, :]   # x_cg - mu_jg
    variances_per_pair = variances[np.newaxis, :, :]               # sigma^2_jg

    # The bracket of the formula, one value per cell, state and gene: (n, N, d)
    log_density_per_gene = (-0.5 * np.log(2 * np.pi * variances_per_pair)
                            - differences ** 2 / (2 * variances_per_pair))

    # Sum over genes: (n, N)
    log_densities = np.sum(log_density_per_gene, axis=2)
    return log_densities


def e_step(X, weeks, number_of_weeks, pi, A, means, variances):
    """
    E-step: responsibilities r_cj = P(z_c = j | x_c, t_c) and the log-likelihood of all cells.

    For each cell c, measured in week t_c, Bayes' rule with the prior p_{t_c}:

        r_cj = p_{t_c}[j] N(x_c; mu_j, U_j) / sum_k p_{t_c}[k] N(x_c; mu_k, U_k)
        l    = sum_c log sum_k p_{t_c}[k] N(x_c; mu_k, U_k)

    computed in logs, in five steps (numbered in the code below):

        1. p_{t_c} = pi A^{t_c}                                  prior of the cell's week
        2. a_cj = log p_{t_c}[j] + log N(x_c; mu_j, U_j)         log of the numerator
        3. l_c  = log sum_k exp(a_ck)                            log of the denominator (log-sum-exp)
        4. r_cj = exp(a_cj - l_c)                                numerator / denominator
        5. l    = sum_c l_c                                      log-likelihood

    Log-sum-exp avoids exp(a_cj) rounding to 0 when there are many genes.

    Parameters
    ----------
    X:         (n, d) expression of each cell
    weeks:     (n,) int in 0..W-1, the week each cell was measured in
    number_of_weeks: W, number of weeks including week 0
    pi:        (N,) initial hidden state proportions (week 0)
    A:         (N, N) one-week transition probabilities
    means:     (N, d) mean expression of each hidden state
    variances: (N, d) variance of each gene in each hidden state (diagonal of U_j)

    Returns
    -------
    r:              (n, N) responsibilities; each row sums to 1
    log_likelihood: float, l = sum over cells of log sum_j p_{t_c}[j] N(x_c; mu_j, U_j)
    """
    # Step 1. Prior of each cell = proportions of its own week, p_{t_c} = pi A^{t_c}: (n, N)
    proportions = get_proportions(pi, A, number_of_weeks)        # (W, N), row t is p_t
    priors = proportions[weeks]                                   # row c is p_{t_c}

    # Step 2. Log of the numerator of r_cj: a_cj = log p_{t_c}[j] + log N(x_c; mu_j, U_j): (n, N)
    log_densities = log_gaussian_densities(X, means, variances)
    log_weights = np.log(priors) + log_densities

    # Step 3. Log of the denominator of r_cj: l_c = log sum_k exp(a_ck): (n,)
    row_logsumexp = logsumexp(log_weights, axis=1)

    # Step 4. Numerator / denominator: r_cj = exp(a_cj - l_c): (n, N)
    r = np.exp(log_weights - row_logsumexp[:, np.newaxis])

    # Step 5. Log-likelihood: l = sum_c l_c
    log_likelihood = np.sum(row_logsumexp)
    return r, log_likelihood


def m_step(X, r, variance_floor=1e-3):
    """
    M-step, part 1: new mean and variance of every hidden state's Gaussian.
    All weeks are pooled, because the Gaussians are shared across weeks.

        mu_jg      = sum_c r_cj x_cg / sum_c r_cj
        sigma^2_jg = sum_c r_cj (x_cg - mu_jg)^2 / sum_c r_cj

    in four steps (numbered in the code below), one hidden state j at a time:

        1. soft_cell_count_j = sum_c r_cj                        how many cells state j "owns"
        2. mu_jg      = sum_c r_cj x_cg / soft_cell_count_j      weighted mean
        3. sigma^2_jg = sum_c r_cj (x_cg - mu_jg)^2 / soft_cell_count_j   weighted variance, new mean
        4. sigma^2_jg = max(sigma^2_jg, variance_floor)          stop a variance collapsing to 0

    Parameters
    ----------
    X:              (n, d) expression of each cell
    r:              (n, N) responsibilities from the E-step; each row sums to 1
    variance_floor: smallest variance allowed

    Returns
    -------
    means:     (N, d) new mean expression of each hidden state
    variances: (N, d) new variance of each gene in each hidden state (diagonal of U_j)
    """
    number_of_states = r.shape[1]
    number_of_genes = X.shape[1]
    means = np.zeros((number_of_states, number_of_genes))
    variances = np.zeros((number_of_states, number_of_genes))

    for j in range(number_of_states):
        r_j = r[:, j]                       # r_cj of every cell c: (n,)
        r_j_column = r_j[:, np.newaxis]     # (n, 1), so that r_cj multiplies every gene of cell c

        # Step 1. Soft number of cells in state j: sum_c r_cj
        soft_cell_count = np.sum(r_j)

        # Step 2. Weighted mean of each gene: (d,)
        means[j] = np.sum(r_j_column * X, axis=0) / soft_cell_count

        # Step 3. Weighted variance of each gene around the new mean: (d,)
        variances[j] = np.sum(r_j_column * (X - means[j]) ** 2, axis=0) / soft_cell_count

    # Step 4. Variance floor
    variances = np.maximum(variances, variance_floor)
    return means, variances


def soft_counts(r, weeks, number_of_weeks):
    """
    Soft number of cells in each hidden state in each week:

        n_t[j] = sum over the cells c of week t of r_cj

    Parameters
    ----------
    r:     (n, N) responsibilities from the E-step
    weeks: (n,) int in 0..W-1, the week each cell was measured in
    number_of_weeks: W, number of weeks including week 0

    Returns
    -------
    counts: (W, N), row t is n_t; row t sums to the number of cells measured in week t
    """
    counts = np.zeros((number_of_weeks, r.shape[1]))
    for t in range(number_of_weeks):
        counts[t] = np.sum(r[weeks == t], axis=0)
    return counts


def expected_log_weights(pi, A, counts):
    """
    The part of the expected log-likelihood that depends on pi and A:

        Q(pi, A) = sum_t sum_j n_t[j] log p_t[j],   p_t = pi A^t

    Parameters
    ----------
    pi:     (N,) initial state proportions
    A:      (N, N) transition probabilities
    counts: (W, N) soft counts n_t[j] from soft_counts

    Returns
    -------
    Q: float
    """
    proportions = get_proportions(pi, A, len(counts))   # (W, N), row t is p_t
    return np.sum(counts * np.log(proportions))


def negative_q_function(theta, counts):
    """
    The function that update_pi_A gives to scipy's minimize: -Q(pi, A) for the pi, A given by the
    free numbers theta, divided by the number of cells so that the optimiser sees numbers near 1.
    Q is the EM Q-function (expected complete-data log-likelihood), here only its pi, A part
    (expected_log_weights). Minimising -Q is the same as maximising Q; scipy can only minimise.

    theta -> pi, A by a softmax, for pi and for each row of A:

        softmax(v)_j = exp(v_j) / sum_k exp(v_k)

    so any N + N^2 numbers give a valid pi and A (non-negative, rows sum to 1).

    Parameters
    ----------
    theta:  (N + N^2,) free numbers: log pi, then the rows of log A one after the other
    counts: (W, N) soft counts n_t[j] from soft_counts

    Returns
    -------
    float, -Q / (number of cells)
    """
    N = counts.shape[1]
    pi = softmax(theta[:N])
    A = softmax(theta[N:].reshape(N, N), axis=1)
    return -expected_log_weights(pi, A, counts) / np.sum(counts)


def update_pi_A(counts, pi, A, max_iterations=50, probability_floor=1e-10):
    """
    M-step, part 2: new pi and A that raise Q. No closed form, because A
    appears as A^2 and A^3, so a numerical optimiser searches for them.

    in five steps (numbered in the code below):

        1. pi, A -> 12 free numbers theta = (log pi, log A)
        2. minimise -Q over theta (L-BFGS-B)                   negative_q_function
        3. theta -> pi, A by a softmax                         any 12 numbers give a valid pi, A
        4. entries below probability_floor -> floor, then rows sum to 1 again
        5. keep the result only if Q went up                   generalised EM: l never decreases

    Step 4 is needed only because of rounding: a softmax is never 0 on paper, but exp(-800) rounds
    to exactly 0 in floating point when an entry of A wants to be 0. Then log 0 = -inf in the next
    Step 1, the optimiser returns nan, Step 5 rejects it, and pi, A stay frozen for the rest of the fit.

    Parameters
    ----------
    counts:            (W, N) soft counts n_t[j] from soft_counts
    pi:                (N,) current initial state proportions
    A:                 (N, N) current transition probabilities
    max_iterations:    int, upper limit on optimiser iterations
    probability_floor: smallest entry allowed in pi and A

    Returns
    -------
    pi: (N,) new initial state proportions
    A:  (N, N) new transition probabilities
    """
    N = len(pi)

    # Step 1. Starting free numbers from the current pi, A: log pi, then the rows of log A one after the other
    theta_start = np.concatenate([np.log(pi), np.log(A).ravel()])

    # Step 2. Minimise -Q over theta.
    # jac="3-point": a numerical gradient precise enough for the flat T and MU rows of A
    result = minimize(
        negative_q_function,
        theta_start,
        args=(counts,),
        method="L-BFGS-B",
        jac="3-point",
        options={"maxiter": max_iterations, "ftol": 1e-15, "gtol": 1e-12},
    )

    # Step 3. Free numbers -> new pi, A: softmax for pi and for each row of A (as in negative_q_function)
    new_pi = softmax(result.x[:N])
    new_A = softmax(result.x[N:].reshape(N, N), axis=1)

    # Step 4. No exact 0 in pi or A (as the floor in householder_start), then rows sum to 1 again
    new_pi = np.maximum(new_pi, probability_floor)
    new_pi = new_pi / new_pi.sum()
    new_A = np.maximum(new_A, probability_floor)
    new_A = new_A / new_A.sum(axis=1, keepdims=True)

    # Step 5. Keep the new pi, A only if Q went up
    if expected_log_weights(new_pi, new_A, counts) > expected_log_weights(pi, A, counts):
        return new_pi, new_A
    return pi, A


def fit_population_model(
    X,
    weeks,
    number_of_weeks,
    pi,
    A,
    means,
    variances,
    max_iterations=500,
    tolerance=1e-6,
    variance_floor=1e-3,
):
    """
    Fit pi, A and the Gaussians to the cells by EM: repeat the E-step and
    the two parts of the M-step until the log-likelihood improves by less than tolerance.

    in seven steps (numbered in the code below):

        0. E-step with the starting values             r, l_0
        then in each iteration k = 1, 2, ...:
        1. Gaussians: mu_j, sigma^2_jg from r          m_step (closed form)
        2. soft counts n_t[j] from r                   soft_counts
        3. pi, A that raise Q(pi, A)                   update_pi_A (numerical)
        4. E-step with the new parameters              r, l_k
        5. check l_k >= l_{k-1}                        EM never lowers l; a drop means a bug
        6. stop if l_k - l_{k-1} < tolerance

    The E-step comes last in each iteration, so log_likelihoods[0] is the l of the starting values
    and log_likelihoods[-1] is the l of the parameters that are returned.

    Parameters
    ----------
    X:         (n, d) expression of each cell
    weeks:     (n,) int in 0..W-1, the week each cell was measured in
    number_of_weeks: W, number of weeks including week 0
    pi:        (N,) starting initial state proportions
    A:         (N, N) starting transition probabilities; no entry exactly 0 (log A is taken)
    means:     (N, d) starting mean expression of each hidden state
    variances: (N, d) starting variances (diagonal of U_j)
    max_iterations: int, upper limit on EM iterations
    tolerance:      float, stop when l improves by less than this
    variance_floor: smallest variance allowed in the Gaussian update

    Returns
    -------
    pi:        (N,) fitted initial state proportions
    A:         (N, N) fitted transition probabilities
    means:     (N, d) fitted mean expression of each hidden state
    variances: (N, d) fitted variances (diagonal of U_j)
    log_likelihoods: list of float, l of the start, then one per iteration; never decreasing
    """
    # Step 0. E-step with the starting values
    r, log_likelihood = e_step(X, weeks, number_of_weeks, pi, A, means, variances)
    log_likelihoods = [log_likelihood]

    for _ in range(max_iterations):
        # Step 1. M-step part 1: new Gaussians, all weeks pooled
        means, variances = m_step(X, r, variance_floor)

        # Step 2. Soft number of cells in each state in each week: (W, N)
        counts = soft_counts(r, weeks, number_of_weeks)

        # Step 3. M-step part 2: new pi, A (kept only if Q went up)
        pi, A = update_pi_A(counts, pi, A)

        # Step 4. E-step with the new parameters
        r, log_likelihood = e_step(X, weeks, number_of_weeks, pi, A, means, variances)
        log_likelihoods.append(log_likelihood)

        # Step 5. l never decreases (up to rounding in the sum over cells)
        improvement = log_likelihoods[-1] - log_likelihoods[-2]
        assert improvement >= -1e-9 * abs(log_likelihoods[-2]), \
            f"log-likelihood went down by {-improvement:.3g} in iteration {len(log_likelihoods) - 1}"

        # Step 6. Converged
        if improvement < tolerance:
            break

    return pi, A, means, variances, log_likelihoods


def match_states(pi, A, means, variances, means_reference):
    """
    Put the fitted states in the order of the reference states. EM numbers its states in an
    arbitrary order (fitted state 0 may be MU), so before comparing a fit with the true parameters,
    fitted state j must be the one whose mean is closest to reference state j.

    in three steps (numbered in the code below):

        1. for every ordering of the N fitted states (N! of them, 6 for N = 3):
           distance = sum_j || mu_fit[order[j]] - mu_ref_j ||^2
        2. keep the ordering with the smallest distance
        3. reorder: new state j = fitted state order[j]
              pi_new[j]     = pi[order[j]]
              A_new[j, k]   = A[order[j], order[k]]      rows and columns, so A_jk stays "from j to k"
              means_new[j]  = means[order[j]],  variances_new[j] = variances[order[j]]

    Reordering only renames the states: the proportions, r and l of the cells do not change.

    Parameters
    ----------
    pi:              (N,) fitted initial state proportions
    A:               (N, N) fitted transition probabilities
    means:           (N, d) fitted means
    variances:       (N, d) fitted variances (diagonal of U_j)
    means_reference: (N, d) means in the wanted order (toy: the true means)

    Returns
    -------
    pi, A, means, variances: the same parameters, with state j matched to reference state j
    """
    number_of_states = len(pi)
    smallest_distance = np.inf

    for order in permutations(range(number_of_states)):
        order = list(order)

        # Step 1. Total squared distance between the reordered fitted means and the reference means
        distance = np.sum((means[order] - means_reference) ** 2)

        # Step 2. Keep the closest ordering
        if distance < smallest_distance:
            smallest_distance = distance
            best_order = order

    # Step 3. Reorder pi, A (rows, then columns), means and variances
    return pi[best_order], A[best_order][:, best_order], means[best_order], variances[best_order]
