"""
Starting values for the population model's EM (report Section 5): a Householder start for A,
random starts for pi and A, and k-means starting Gaussians. Shared by the toy and the real data.
Shapes follow the conventions at the top of population_model.py.
"""
import numpy as np
from sklearn.cluster import KMeans


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

    The clusters come in arbitrary order; Step 9 (match_states) puts them in the order MA, T, MU.

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
    cluster_proportions = np.zeros((number_of_weeks, number_of_states))
    for t in range(number_of_weeks):
        labels_in_week = cluster_labels[weeks == t]
        cluster_proportions[t] = np.bincount(labels_in_week, minlength=number_of_states) / len(labels_in_week)

    return means, variances, cluster_proportions
