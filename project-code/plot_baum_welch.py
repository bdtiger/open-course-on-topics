"""
Figure for the progress report: what Baum-Welch does on the toy data.

Left panel:  log-likelihood per iteration for several random starts, against the
             value under the true parameters. Every curve rises monotonically, the
             curves end at different heights (local optima), and the best one ends
             slightly above the true-parameter value.
Right panel: the emission matrix B. True B, the fitted B from the best start as it
             comes out of Baum-Welch (rows in a different order: label switching),
             and the same fitted B after matching the state labels.

Run:  ../.venv/bin/python plot_baum_welch.py
Writes ../progress-report/figures/baum_welch.pdf
"""
from itertools import permutations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from hmm import pi, A, B, STATES, SYMBOLS, sample_hmm_sequence, forward_algorithm, baum_welch

N_RESTARTS = 5
T = 2000

# ---- run the experiment ----------------------------------------------------
_, observations = sample_hmm_sequence(pi, A, B, T, np.random.default_rng(0))
_, _, log_likelihood_true = forward_algorithm(pi, A, B, observations)

runs = []
for seed in range(1, N_RESTARTS + 1):
    rng = np.random.default_rng(seed)
    pi_guess = rng.dirichlet(np.ones(len(STATES)))
    A_guess = rng.dirichlet(np.ones(len(STATES)), size=len(STATES))
    B_guess = rng.dirichlet(np.ones(len(SYMBOLS)), size=len(STATES))
    pi_fit, A_fit, B_fit, log_likelihoods = baum_welch(pi_guess, A_guess, B_guess, observations, max_iterations=200)
    assert np.all(np.diff(log_likelihoods) >= -1e-8)
    runs.append(dict(seed=seed, B=B_fit, ll=np.array(log_likelihoods)))

best = max(runs, key=lambda r: r["ll"][-1])

# Panel (b) shows one start from the top plateau whose states came out in a different order.
# Start 4 reaches the same log-likelihood as the best start (within 1) and its rows are permuted.
SHOWN_SEED = 4
shown = next(r for r in runs if r["seed"] == SHOWN_SEED)

# match labels: the row order of the fitted B that is closest to the true B
shown_perm = min(permutations(range(len(STATES))),
                 key=lambda p: np.abs(shown["B"][list(p)] - B).sum())
B_matched = shown["B"][list(shown_perm)]

# ---- figure ----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.edgecolor": "#b3b2ad", "axes.linewidth": 0.5,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
})
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed categorical order
GRAY = "#52514e"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#2a78d6", "#0d3b73"])

fig = plt.figure(figsize=(7.4, 2.5))
grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.1], wspace=0.3)

# (a) log-likelihood curves
ax = fig.add_subplot(grid[0])
ax.axhline(log_likelihood_true, color=GRAY, lw=1, ls="--", zorder=1)
ax.text(0.98, log_likelihood_true + 8, f"true parameters ({log_likelihood_true:.0f})",
        transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color=GRAY)
for color, run in zip(SERIES, runs):
    ll = run["ll"]
    ax.plot(np.arange(1, len(ll) + 1), ll, color=color, lw=1.4,
            label=f"start {run['seed']}" + (" (best)" if run is best else ""), zorder=2)
ax.set_xlim(0, 60)
ax.set_ylim(log_likelihood_true - 250, log_likelihood_true + 40)
ax.set_xlabel("Baum–Welch iteration")
ax.set_ylabel(r"$\log P(O \mid \lambda)$")
ax.set_title("(a) Log-likelihood, 5 random starts", loc="left", fontsize=8, fontweight="bold")
ax.grid(axis="y", color="#e7e6e1", lw=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=6.5, loc="lower right", handlelength=1.4, borderaxespad=0.2)

# (b) emission matrices
sub = grid[1].subgridspec(1, 3, wspace=0.45)
panels = [("true $B$\n(generating)", B), (f"fitted $B$, start {shown['seed']}\n(as returned)", shown["B"]),
          (f"fitted $B$, start {shown['seed']}\n(labels matched)", B_matched)]
for k, (title, mat) in enumerate(panels):
    axb = fig.add_subplot(sub[k])
    axb.imshow(mat, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axb.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.8,
                     color="white" if mat[i, j] > 0.5 else "#0b0b0b")
    axb.set_xticks(range(len(SYMBOLS)), SYMBOLS)
    axb.set_yticks(range(len(STATES)), STATES if k != 1 else [f"{i+1}" for i in range(len(STATES))])
    axb.tick_params(length=0)
    for s in axb.spines.values():
        s.set_visible(False)
    axb.set_title(title, fontsize=7.5)
    if k == 0:
        axb.set_ylabel("state")
    elif k == 1:
        axb.set_ylabel("fitted state")
    axb.set_xlabel("symbol")
fig.text(0.67, 1.0, "(b) Emission matrix $B$: rows are states, columns are symbols",
         ha="center", va="bottom", fontsize=8, fontweight="bold")

out = Path(__file__).resolve().parent.parent / "progress-report" / "figures" / "baum_welch.pdf"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
print("saved", out)
print("true-parameter log-likelihood:", round(log_likelihood_true, 1))
for r in runs:
    print(f"start {r['seed']}: {len(r['ll'])} iterations, final log-likelihood {r['ll'][-1]:.1f}")
print("best start:", best["seed"])
print(f"start {shown['seed']} shown in panel (b); its rows in the order that matches the true B:", [s + 1 for s in shown_perm])
