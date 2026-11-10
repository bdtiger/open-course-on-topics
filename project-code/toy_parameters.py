"""
True parameters of the toy population model (2 genes, 3 hidden states, 4 weeks).

Constants only: no functions, no prints. The simulator is the shared population_model.sample_cells.
Shapes follow the conventions at the top of population_model.py.
"""
import numpy as np

from population_model import validate_population_parameters


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


validate_population_parameters(pi, A, means, variances)
