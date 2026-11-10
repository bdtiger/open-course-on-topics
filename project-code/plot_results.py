"""
Figures for the report, made from the saved results in results/ (see utils.py).
This file never runs an experiment. To refresh a figure: first run_experiments.py, then this file.

One function per panel (takes a results dict and an axis), one function per figure (composes panels
and saves the file), and write_numbers_tex, which writes the numbers the report quotes as LaTeX macros.

Run:  ../.venv/bin/python plot_results.py
"""
from itertools import permutations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils import load_results

FIGURES_DIR = Path(__file__).resolve().parent.parent / "progress-report" / "figures"

# Colours. Categorical: fixed order, one per restart. Magnitude: one hue, light to dark.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#52514e"
GRID = "#e7e6e1"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#2a78d6", "#0d3b73"])


def apply_style():
    """Set the matplotlib defaults used by every figure (serif font, small text, light axes)."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.edgecolor": "#b3b2ad", "axes.linewidth": 0.5,
        "xtick.color": GRAY, "ytick.color": GRAY,
        "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    })


def save_figure(fig, name):
    """
    Save a figure as figures/<name>.pdf (for the report) and figures/<name>.png (to look at).

    Parameters
    ----------
    fig: matplotlib Figure
    name: str, file name without extension

    Returns
    -------
    pathlib.Path of the pdf written
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved", path)
    return path


# ---------------------------------------------------------------- helpers

def best_restart(results):
    """
    The restart with the highest final log-likelihood.

    Parameters
    ----------
    results: dict from results/baum_welch_restarts.json

    Returns
    -------
    dict, one entry of results["restarts"]
    """
    return max(results["restarts"], key=lambda r: r["log_likelihoods"][-1])


def match_state_order(B_fit, B_true):
    """
    Find the row order of a fitted B that best matches the true B (undo label switching).

    Parameters
    ----------
    B_fit: (N, M) fitted emission matrix
    B_true: (N, M) true emission matrix

    Returns
    -------
    list of int, length N; B_fit[order] is the re-ordered matrix closest to B_true
    """
    B_fit, B_true = np.asarray(B_fit), np.asarray(B_true)
    return list(min(permutations(range(len(B_true))),
                    key=lambda order: np.abs(B_fit[list(order)] - B_true).sum()))


def restart_to_show(results):
    """
    Pick the restart for the emission-matrix panel: the best restart whose states came out in a
    different order than the true ones, so the panel shows label switching. Falls back to the best restart.

    Parameters
    ----------
    results: dict from results/baum_welch_restarts.json

    Returns
    -------
    dict, one entry of results["restarts"]
    """
    ranked = sorted(results["restarts"], key=lambda r: -r["log_likelihoods"][-1])
    for r in ranked:
        if match_state_order(r["B"], results["true"]["B"]) != list(range(len(results["states"]))):
            return r
    return ranked[0]


# ---------------------------------------------------------------- panels

def plot_log_likelihood_curves(results, ax):
    """
    Panel: log-likelihood per iteration, one line per restart, dashed line at the true-parameter value.

    Parameters
    ----------
    results: dict from results/baum_welch_restarts.json
    ax: matplotlib Axes to draw on

    Returns
    -------
    ax
    """
    ll_true = results["true"]["log_likelihood"]
    ax.axhline(ll_true, color=GRAY, lw=1, ls="--", zorder=1)
    ax.text(0.98, ll_true + 8, f"true parameters ({ll_true:.0f})".replace("-", "\u2212"),
            transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color=GRAY)
    best = best_restart(results)
    for color, r in zip(SERIES, results["restarts"]):
        ll = r["log_likelihoods"]
        label = f"start {r['restart']}" + (" (best)" if r is best else "")
        ax.plot(np.arange(1, len(ll) + 1), ll, color=color, lw=1.4, label=label, zorder=2)
    ax.set_xlim(0, 60)
    ax.set_ylim(ll_true - 250, ll_true + 40)
    ax.set_xlabel("Baum–Welch iteration")
    ax.set_ylabel(r"$\log P(O \mid \lambda)$")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right", handlelength=1.4, borderaxespad=0.2)
    return ax


def plot_matrix(matrix, row_labels, col_labels, ax, title=""):
    """
    Panel: one probability matrix as a heatmap with the value written in every cell.

    Parameters
    ----------
    matrix: (R, C) values in [0, 1]
    row_labels: list of str, length R
    col_labels: list of str, length C
    ax: matplotlib Axes to draw on
    title: str, optional

    Returns
    -------
    ax
    """
    matrix = np.asarray(matrix)
    ax.imshow(matrix, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=5.8,
                    color="white" if matrix[i, j] > 0.5 else "#0b0b0b")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=7.5)
    return ax


def plot_emission_comparison(results, restart, axes):
    """
    Panel: true B, the fitted B of one restart as returned, and the same fitted B with rows re-ordered
    to match the true states.

    Parameters
    ----------
    results: dict from results/baum_welch_restarts.json
    restart: dict, one entry of results["restarts"]
    axes: list of three matplotlib Axes

    Returns
    -------
    axes
    """
    states, symbols = results["states"], results["symbols"]
    B_true, B_fit = results["true"]["B"], np.asarray(restart["B"])
    order = match_state_order(B_fit, B_true)
    k = restart["restart"]
    plot_matrix(B_true, states, symbols, axes[0], "true $B$\n(generating)")
    plot_matrix(B_fit, [str(i + 1) for i in range(len(states))], symbols, axes[1], f"fitted $B$, start {k}\n(as returned)")
    plot_matrix(B_fit[order], states, symbols, axes[2], f"fitted $B$, start {k}\n(labels matched)")
    axes[0].set_ylabel("state")
    axes[1].set_ylabel("fitted state")
    for ax in axes:
        ax.set_xlabel("symbol")
    return axes


# ---------------------------------------------------------------- figures

def figure_baum_welch(results):
    """
    Report figure: (a) log-likelihood curves, (b) emission matrices. Saves figures/baum_welch.pdf.

    Parameters
    ----------
    results: dict from results/baum_welch_restarts.json

    Returns
    -------
    pathlib.Path of the pdf written
    """
    apply_style()
    fig = plt.figure(figsize=(7.4, 2.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.1], wspace=0.3)

    ax = fig.add_subplot(grid[0])
    plot_log_likelihood_curves(results, ax)
    ax.set_title(f"Log-likelihood, {results['n_restarts']} random starts", loc="left", fontsize=8, fontweight="bold")

    # sub = grid[1].subgridspec(1, 3, wspace=0.45)
    # axes = [fig.add_subplot(sub[i]) for i in range(3)]
    # plot_emission_comparison(results, restart_to_show(results), axes)
    # fig.text(0.67, 1.0, "(b) Emission matrix $B$: rows are states, columns are symbols",
    #          ha="center", va="bottom", fontsize=8, fontweight="bold")
    return save_figure(fig, "baum_welch")


def write_numbers_tex(results):
    """
    Write figures/numbers.tex: the numbers the report quotes, as LaTeX macros, so text and figure
    always come from the same results file. Use in the report with \\input{figures/numbers}.

    Parameters
    ----------
    results: dict from results/baum_welch_restarts.json

    Returns
    -------
    pathlib.Path of the file written
    """
    best, worst = best_restart(results), min(results["restarts"], key=lambda r: r["log_likelihoods"][-1])
    macros = {
        "nRestarts": results["n_restarts"],
        "maxIterations": results["max_iterations"],
        "sequenceLength": results["T"],
        "llTrue": f"{results['true']['log_likelihood']:.1f}",
        "llBest": f"{best['log_likelihoods'][-1]:.1f}",
        "llWorst": f"{worst['log_likelihoods'][-1]:.1f}",
        "bestRestart": best["restart"],
        "worstRestart": worst["restart"],
        "shownRestart": restart_to_show(results)["restart"],
    }
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / "numbers.tex"
    with open(path, "w") as f:
        f.write("% generated by plot_results.py from results/baum_welch_restarts.json; do not edit by hand\n")
        for name, value in macros.items():
            f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")
    print("saved", path)
    return path


if __name__ == "__main__":
    results = load_results("baum_welch_restarts")
    figure_baum_welch(results)
    write_numbers_tex(results)
