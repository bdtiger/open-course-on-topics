
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.cluster import KMeans

# Shape conventions
#   N = number of hidden states (A is N x N)
#   d = number of genes      (each cell is a vector of length d)
#   W = number of weeks      (4: day 0, 7, 14, 21)
#   pi: (N,)   A: (N, N)   proportions: (W, N), row t is p_t
#   means: (N, d)   
#   variances: (N, d), the diagonal of U_j


NUMBER_OF_STATES = 3  # Number of hidden states
NUMBER_OF_GENES = 2  # Number of genes
STATES = ["MA", "T", "MU"]
NUMBER_OF_WEEKS = 4

pi = np.array([0.90, 0.05, 0.05])  # Initial hidden state proportions (day 0)

A = np.array([
    [0.80, 0.15, 0.05],  # Transition probabilities from MA (one week)
    [0.10, 0.60, 0.30],  # Transition probabilities from T
    [0.02, 0.03, 0.95]   # Transition probabilities from MU
])

means = np.array([
    [4.0, 1.0],  # Mean expression of MA
    [2.5, 2.5],  # Mean expression of T
    [1.0, 4.0]   # Mean expression of MU
])

variances = np.ones_like(means)  # Diagonal of U_j, all variances 1


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


validate_population_parameters(pi, A, means, variances)


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


def sample_cells(pi, A, means, variances, cells_per_week, random_generator):
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
    random_generator: numpy Generator; same seed gives the same cells

    Returns
    -------
    X:      (n, d) expression of each cell, n = W * cells_per_week
    weeks:  (n,) int in 0..W-1, the week each cell was measured in
    states: (n,) int in 0..N-1, the true hidden state of each cell (answer key, not seen by the fit)
    """
    number_of_states, number_of_genes = means.shape
    proportions = get_proportions(pi, A, NUMBER_OF_WEEKS)

    weeks = np.repeat(np.arange(NUMBER_OF_WEEKS), cells_per_week)
    states = np.zeros(len(weeks), dtype=int)
    for t in range(NUMBER_OF_WEEKS):
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


def e_step(X, weeks, pi, A, means, variances):
    """
    E-step: responsibilities r_cj = P(z_c = j | x_c, t_c) and the log-likelihood of all cells.

    For each cell c, measured in week t_c, Bayes' rule with the prior p_{t_c} (report Section 3):

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
    proportions = get_proportions(pi, A, NUMBER_OF_WEEKS)        # (W, N), row t is p_t
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
    M-step, part 1: new mean and variance of every hidden state's Gaussian (report Section 4).
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


def soft_counts(r, weeks):
    """
    Soft number of cells in each hidden state in each week (report Section 4):

        n_t[j] = sum over the cells c of week t of r_cj

    Parameters
    ----------
    r:     (n, N) responsibilities from the E-step
    weeks: (n,) int in 0..W-1, the week each cell was measured in

    Returns
    -------
    counts: (W, N), row t is n_t; row t sums to the number of cells measured in week t
    """
    counts = np.zeros((NUMBER_OF_WEEKS, r.shape[1]))
    for t in range(NUMBER_OF_WEEKS):
        counts[t] = np.sum(r[weeks == t], axis=0)
    return counts


def expected_log_weights(pi, A, counts):
    """
    The part of the expected log-likelihood that depends on pi and A (report Section 4):

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


def update_pi_A(counts, pi, A, max_iterations=50):
    """
    M-step, part 2: new pi and A that raise Q (report Section 4). No closed form, because A
    appears as A^2 and A^3, so a numerical optimiser searches for them.

    in four steps (numbered in the code below):

        1. pi, A -> 12 free numbers theta = (log pi, log A)    start from the current pi, A
        2. theta -> pi, A by a softmax                           any 12 numbers give a valid pi, A
        3. minimise -Q over theta (L-BFGS-B)
        4. keep the result only if Q went up                     generalised EM: l never decreases

    Parameters
    ----------
    counts:         (W, N) soft counts n_t[j] from soft_counts
    pi:             (N,) current initial state proportions
    A:              (N, N) current transition probabilities
    max_iterations: int, upper limit on optimiser iterations

    Returns
    -------
    pi: (N,) new initial state proportions
    A:  (N, N) new transition probabilities
    """
    N = len(pi)

    # Step 1. Free numbers: log pi, then the rows of log A one after the other
    theta_start = np.concatenate([np.log(pi), np.log(A).ravel()])

    # Step 2. Free numbers -> pi, A: softmax(v)_j = exp(v_j) / sum_k exp(v_k), for pi and for each row of A
    def to_pi_A(theta):
        return softmax(theta[:N]), softmax(theta[N:].reshape(N, N), axis=1)

    # Step 3. Minimise -Q. Divided by the number of cells, so the optimiser sees numbers near 1.
    # jac="3-point": a numerical gradient precise enough for the flat T and MU rows of A
    def negative_Q(theta):
        new_pi, new_A = to_pi_A(theta)
        return -expected_log_weights(new_pi, new_A, counts) / np.sum(counts)

    result = minimize(negative_Q, theta_start, method="L-BFGS-B", jac="3-point",
                      options={"maxiter": max_iterations, "ftol": 1e-15, "gtol": 1e-12})

    # Step 4. Keep the new pi, A only if Q went up
    if negative_Q(result.x) < negative_Q(theta_start):
        return to_pi_A(result.x)
    return pi, A

def householder_start(p0, p1, floor=0.01):
    """
    Starting A from a Householder reflection of rough day-0 and day-7 proportions (report Section 5):

        A_start = I - 2 w w^T / (w^T w),   w = p0 - p1

    in five steps (numbered in the code below):

        1. w = p0 - p1                       sums to 0, because p0 and p1 both sum to 1
        2. reflection = 2 w w^T / (w^T w)    (N, N)
        3. A_start = I - reflection          every row sums to 1, but entries can be negative
        4. entries below floor -> floor      no negative entry, and no entry exactly 0
        5. divide each row by its sum        rows sum to 1 again

    Parameters
    ----------
    p0:    (N,) rough state proportions on day 0
    p1:    (N,) rough state proportions on day 7
    floor: smallest entry allowed in A_start

    Returns
    -------
    A_start: (N, N) starting transition probabilities; rows sum to 1, every entry > 0
    """
    # Step 1. Difference of the two proportion vectors
    w = p0 - p1
    assert w @ w > 0, "p0 and p1 are equal, so the reflection is undefined; use a random start"

    # Step 2. 2 w w^T / (w^T w): outer product of w with itself, divided by its squared length
    reflection = 2 * np.outer(w, w) / (w @ w)

    # Step 3. Householder matrix
    A_start = np.eye(len(w)) - reflection

    # Step 4. Replace negative entries (and anything below the floor) by the floor
    A_start = np.maximum(A_start, floor)

    # Step 5. Make every row a probability distribution again
    A_start = A_start / A_start.sum(axis=1, keepdims=True)
    return A_start


def random_start(number_of_states, random_generator):
    """
    Random starting pi and A: every row a random probability distribution (uniform Dirichlet),
    as random_parameters in hmm.py. A Dirichlet draw is never exactly 0.

    Parameters
    ----------
    number_of_states: N
    random_generator: numpy Generator; different seed gives a different start

    Returns
    -------
    pi: (N,) random initial state proportions
    A:  (N, N) random transition probabilities, rows sum to 1
    """
    pi = random_generator.dirichlet(np.ones(number_of_states))
    A = random_generator.dirichlet(np.ones(number_of_states), size=number_of_states)
    return pi, A


def initial_gaussians(X, weeks, number_of_states, random_generator, variance_floor=1e-3):
    """
    Starting Gaussians from k-means on all cells (all weeks pooled), plus the fraction of each
    week's cells in each cluster. Rows 0 and 1 of those fractions are the rough p0, p1 for
    householder_start.

    in four steps (numbered in the code below):

        1. k-means with N clusters on all cells       each cell gets a cluster label 0..N-1
        2. mu_j = centre of cluster j
        3. sigma^2_jg = variance of gene g inside cluster j (floored)
        4. cluster_proportions[t, j] = fraction of week-t cells in cluster j

    The clusters come in arbitrary order; Step 9 (match_states) puts them in the order MA, T, MU.

    Parameters
    ----------
    X:                (n, d) expression of each cell
    weeks:            (n,) int in 0..W-1, the week each cell was measured in
    number_of_states: N, number of clusters
    random_generator: numpy Generator; seeds k-means
    variance_floor:   smallest variance allowed (a cluster can be very tight)

    Returns
    -------
    means:               (N, d) cluster centres
    variances:           (N, d) within-cluster variance of each gene
    cluster_proportions: (W, N) row t is the fraction of week-t cells in each cluster
    """
    number_of_genes = X.shape[1]

    # Step 1. k-means on all cells
    kmeans = KMeans(n_clusters=number_of_states, n_init=10,
                    random_state=int(random_generator.integers(1_000_000)))
    cluster_labels = kmeans.fit_predict(X)   # (n,) cluster of each cell

    # Step 2. Means = cluster centres
    means = kmeans.cluster_centers_

    # Step 3. Variances = spread of each gene inside each cluster
    variances = np.zeros((number_of_states, number_of_genes))
    for j in range(number_of_states):
        variances[j] = X[cluster_labels == j].var(axis=0)
    variances = np.maximum(variances, variance_floor)

    # Step 4. Fraction of each week's cells in each cluster
    cluster_proportions = np.zeros((NUMBER_OF_WEEKS, number_of_states))
    for t in range(NUMBER_OF_WEEKS):
        labels_in_week = cluster_labels[weeks == t]
        cluster_proportions[t] = np.bincount(labels_in_week, minlength=number_of_states) / len(labels_in_week)

    return means, variances, cluster_proportions



if __name__ == "__main__":
    # Quick sanity check: proportions sum to 1, sample matches the parameters it was drawn from
    proportions = get_proportions(pi, A, NUMBER_OF_WEEKS)
    print("proportions (rows are p_0 .. p_3):\n", proportions)

    cells_per_week = 3
    X, weeks, states = sample_cells(
        pi, 
        A, 
        means, 
        variances, 
        cells_per_week,
        np.random.default_rng(0)
    )

    print(f"Sampled {len(X)} cells, {cells_per_week} per week.\n")
    print(f"size of Weeks:{weeks.shape}\n", weeks)
    print(f"size of States:{states.shape}\n", states)
