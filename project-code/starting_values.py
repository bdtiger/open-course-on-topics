"""
Starting values for the population model's EM: a Householder start for A,
random starts for pi and A, k-means starting Gaussians, and fit_from_several_starts, which fits
from all of them and keeps the best. Shared by the toy and the real data.
Shapes follow the conventions at the top of population_model.py.
"""
import numpy as np
from sklearn.cluster import KMeans

from population_model import fit_population_model


def householder_start(p0, p1, floor=0.01):
    """
    Starting A from a Householder reflection of rough day-0 and day-7 proportions:

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


def initial_gaussians(X, weeks, number_of_states, number_of_weeks, random_generator, variance_floor=1e-3):
    """
    Starting Gaussians from k-means on all cells (all weeks pooled), plus the fraction of each
    week's cells in each cluster. Rows 0 and 1 of those fractions are the rough p0, p1 for
    householder_start.

    in four steps (numbered in the code below):

        1. k-means with N clusters on all cells       each cell gets a cluster label 0..N-1
        2. mu_j = centre of cluster j
        3. sigma^2_jg = variance of gene g inside cluster j (floored)
        4. cluster_proportions[t, j] = fraction of week-t cells in cluster j

    The clusters come in arbitrary order; match_states (population_model.py) puts them in the order MA, T, MU.

    Parameters
    ----------
    X:                (n, d) expression of each cell
    weeks:            (n,) int in 0..W-1, the week each cell was measured in
    number_of_states: N, number of clusters
    number_of_weeks:  W, number of weeks including week 0
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
    kmeans = KMeans(
        n_clusters=number_of_states,
        n_init=10,
        random_state=int(random_generator.integers(1_000_000)),
    )
    cluster_labels = kmeans.fit_predict(X)   # (n,) cluster of each cell

    # Step 2. Means = cluster centres
    means = kmeans.cluster_centers_

    # Step 3. Variances = spread of each gene inside each cluster
    variances = np.zeros((number_of_states, number_of_genes))
    for j in range(number_of_states):
        variances[j] = X[cluster_labels == j].var(axis=0)
    variances = np.maximum(variances, variance_floor)

    # Step 4. Fraction of each week's cells in each cluster
    cluster_proportions = np.zeros((number_of_weeks, number_of_states))
    for t in range(number_of_weeks):
        labels_in_week = cluster_labels[weeks == t]
        cluster_proportions[t] = np.bincount(labels_in_week, minlength=number_of_states) / len(labels_in_week)

    return means, variances, cluster_proportions


def fit_from_several_starts(
    X,
    weeks,
    number_of_states,
    number_of_weeks,
    number_of_random_starts,
    random_generator,
    max_iterations=500,
):
    """
    Fit the population model by EM from several starting values and keep every fit. EM only climbs
    to the nearest local maximum of l, so the answer is the fit with the highest final l. One start
    is not enough: on the toy data, the k-means start gets stuck with MA split in two.

    in four steps (numbered in the code below):

        1. Householder start: k-means Gaussians, pi = week-0 cluster fractions,
           A = householder_start(week-0 cluster fractions, week-1 cluster fractions)
        2. random starts, each: random pi, A (random_start), means = N random cells,
           variances = variance of each gene over all cells
        3. fit_population_model from every start
        4. best = the fit with the highest final l

    The random starts draw their own means, not the k-means centres: with the same k-means
    Gaussians every time, the random pi, A all ended in the same few local maxima (toy trial,
    3000 cells per week).

    Parameters
    ----------
    X:                       (n, d) expression of each cell
    weeks:                   (n,) int in 0..W-1, the week each cell was measured in
    number_of_states:        N
    number_of_weeks:         W, number of weeks including week 0
    number_of_random_starts: int, number of random starts besides the Householder start
    random_generator:        numpy Generator; seeds k-means and every random start
    max_iterations:          int, upper limit on EM iterations per start

    Returns
    -------
    fits:       list of dicts, one per start (the Householder start first), each with keys
                "start" ("householder" or "random"), "pi", "A", "means", "variances" (fitted,
                states in arbitrary order) and "log_likelihoods" (l of the start, then one per iteration)
    best_index: int, position in fits of the fit with the highest final l
    """
    # Step 1. Householder start from k-means on all cells
    means_kmeans, variances_kmeans, cluster_proportions = initial_gaussians(
        X,
        weeks,
        number_of_states,
        number_of_weeks,
        random_generator,
    )
    A_householder = householder_start(cluster_proportions[0], cluster_proportions[1])
    starts = [("householder", cluster_proportions[0], A_householder, means_kmeans, variances_kmeans)]

    # Step 2. Random starts: random pi, A and N different random cells as the means
    for _ in range(number_of_random_starts):
        pi_random, A_random = random_start(number_of_states, random_generator)
        means_random = X[random_generator.choice(len(X), size=number_of_states, replace=False)]
        variances_random = np.tile(X.var(axis=0), (number_of_states, 1))   # (N, d), every row the spread of all cells
        starts.append(("random", pi_random, A_random, means_random, variances_random))

    # Step 3. EM from every start
    fits = []
    for start, pi, A, means, variances in starts:
        pi, A, means, variances, log_likelihoods = fit_population_model(
            X,
            weeks,
            number_of_weeks,
            pi,
            A,
            means,
            variances,
            max_iterations=max_iterations,
        )
        fits.append({
            "start": start,
            "pi": pi,
            "A": A,
            "means": means,
            "variances": variances,
            "log_likelihoods": log_likelihoods,
        })

    # Step 4. Best fit = highest final l
    final_log_likelihoods = [fit["log_likelihoods"][-1] for fit in fits]
    best_index = int(np.argmax(final_log_likelihoods))
    return fits, best_index
