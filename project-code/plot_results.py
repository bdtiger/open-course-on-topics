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
    best_order = min(
        permutations(range(len(B_true))),
        key=lambda order: np.abs(B_fit[list(order)] - B_true).sum(),
    )
    return list(best_order)


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
    ax.text(
        0.98,
        ll_true + 8,
        f"true parameters ({ll_true:.0f})".replace("-", "\u2212"),
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7,
        color=GRAY,
    )
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
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=5.8,
                color="white" if matrix[i, j] > 0.5 else "#0b0b0b",
            )
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


# ---------------------------------------------------------------- panels: toy population model
# Colours: state j always has SERIES[j] (MA blue, T orange, MU green), in every figure.

def plot_cells_of_week(results, week, ax):
    """
    Panel: the simulated cells of one week in the plane of gene 1 and gene 2, coloured by their true
    state, with the true state means as black crosses.

    Parameters
    ----------
    results: dict from results/population_recovery_<cells_per_week>.json
    week: int in 0..W-1
    ax: matplotlib Axes to draw on

    Returns
    -------
    ax
    """
    X = np.asarray(results["cells"]["X"])
    weeks = np.asarray(results["cells"]["weeks"])
    states = np.asarray(results["cells"]["states"])
    for j, state in enumerate(results["states"]):
        cells = X[(weeks == week) & (states == j)]
        ax.scatter(cells[:, 0], cells[:, 1], s=5, color=SERIES[j], alpha=0.75, linewidths=0, label=state)
    means = np.asarray(results["true"]["means"])
    ax.scatter(means[:, 0], means[:, 1], marker="+", s=45, color="#0b0b0b", linewidths=1, zorder=3)
    ax.set_title(f"day {7 * week}", fontsize=8)
    ax.set_xlabel("gene 1 (myeloid)")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_population_log_likelihood_curves(results, ax):
    """
    Panel: log-likelihood per EM iteration for every start. The best start in blue, the Householder
    start in orange, the other random starts in gray; dashed line at l of the true parameters.
    The x-axis is logarithmic (most of the climb is in the first 20 iterations, the slowest start
    runs 2000). The y-axis shows the range of the final values, so the first iterations are below it.

    Parameters
    ----------
    results: dict from results/population_recovery_<cells_per_week>.json
    ax: matplotlib Axes to draw on

    Returns
    -------
    ax
    """
    log_likelihood_true = results["true"]["log_likelihood"]
    final_log_likelihoods = [fit["log_likelihoods"][-1] for fit in results["fits"]]
    best, worst = max(final_log_likelihoods), min(final_log_likelihoods)

    gray_in_legend = False
    for k, fit in enumerate(results["fits"]):
        log_likelihoods = fit["log_likelihoods"][1:]            # drop l of the starting values (iteration 0)
        iterations = np.arange(1, len(log_likelihoods) + 1)     # 1, 2, ... so the log axis can show them
        if k == results["best_index"]:
            ax.plot(iterations, log_likelihoods, color=SERIES[0], lw=1.4, zorder=3, label=f"best (random start {k})")
        elif fit["start"] == "householder":
            ax.plot(iterations, log_likelihoods, color=SERIES[1], lw=1.4, zorder=2, label="Householder start")
        else:
            ax.plot(
                iterations,
                log_likelihoods,
                color="#b3b2ad",
                lw=0.7,
                zorder=1,
                label="_nolegend_" if gray_in_legend else "other random starts",
            )
            gray_in_legend = True

    ax.axhline(log_likelihood_true, color=GRAY, lw=1, ls="--", zorder=1, label="true parameters")
    spread = best - worst
    ax.set_ylim(worst - 0.15 * spread, best + 0.08 * spread)
    ax.set_xscale("log")
    ax.set_xlabel("EM iteration (log scale)")
    ax.set_ylabel(r"log-likelihood $\ell$")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_proportions(results, ax):
    """
    Panel: state proportions per week. Solid lines: p_t = pi A^t of the best fit. Dots: the observed
    soft proportions n_t / (cells in week t) from the best fit's r. Dotted lines: p_t of the true parameters.

    Parameters
    ----------
    results: dict from results/population_recovery_<cells_per_week>.json
    ax: matplotlib Axes to draw on

    Returns
    -------
    ax
    """
    days = 7 * np.arange(results["number_of_weeks"])
    proportions_true = np.asarray(results["true"]["proportions"])
    proportions_fit = np.asarray(results["best"]["proportions"])
    soft_proportions = np.asarray(results["best"]["soft_proportions"])
    for j, state in enumerate(results["states"]):
        ax.plot(days, proportions_true[:, j], color=SERIES[j], lw=1, ls=":", zorder=1)
        ax.plot(days, proportions_fit[:, j], color=SERIES[j], lw=1.4, zorder=2)
        ax.scatter(days, soft_proportions[:, j], s=18, color=SERIES[j], edgecolors="white", linewidths=0.8, zorder=3)
        ax.text(days[-1] + 1, proportions_fit[-1, j], state, va="center", fontsize=7)

    # Legend for the three kinds of line, in gray (the colour is the state)
    ax.plot([], [], color=GRAY, lw=1.4, label=r"fitted $p_t = \hat\pi \hat A^t$")
    ax.scatter([], [], s=18, color=GRAY, label="observed soft proportions")
    ax.plot([], [], color=GRAY, lw=1, ls=":", label=r"true $p_t$")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.6, borderaxespad=0.2)

    ax.set_xticks(days, [f"day {day}" for day in days])
    ax.set_xlim(days[0] - 1.5, days[-1] + 4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("proportion of cells")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


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


def figure_population_cells(results):
    """
    Report figure: the simulated cells of each week, coloured by true state. Shows the population
    moving from MA towards T and MU. Saves figures/population_cells_<cells_per_week>.pdf.

    Parameters
    ----------
    results: dict from results/population_recovery_<cells_per_week>.json

    Returns
    -------
    pathlib.Path of the pdf written
    """
    apply_style()
    number_of_weeks = results["number_of_weeks"]
    fig, axes = plt.subplots(1, number_of_weeks, figsize=(7.2, 2.1), sharex=True, sharey=True)
    for week in range(number_of_weeks):
        plot_cells_of_week(results, week, axes[week])
    axes[0].set_ylabel("gene 2 (mural)")
    axes[0].legend(
        frameon=False,
        fontsize=6.5,
        markerscale=2.5,
        loc="upper right",
        handletextpad=0.1,
        borderaxespad=0.1,
    )
    return save_figure(fig, f"population_cells_{results['cells_per_week']}")


def figure_population_log_likelihood(results_small, results_large):
    """
    Report figure: log-likelihood curves of every start, one panel per number of cells per week.
    Saves figures/population_log_likelihood.pdf.

    Parameters
    ----------
    results_small: dict from results/population_recovery_300.json
    results_large: dict from results/population_recovery_3000.json

    Returns
    -------
    pathlib.Path of the pdf written
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax, results in zip(axes, [results_small, results_large]):
        plot_population_log_likelihood_curves(results, ax)
        ax.set_title(f"{results['cells_per_week']} cells per week", loc="left", fontsize=8)
        # Legend of each panel below it: the best start differs between the panels
        ax.legend(
            frameon=False,
            fontsize=6.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=2,
            handlelength=1.6,
            columnspacing=1.2,
        )
    fig.subplots_adjust(wspace=0.3)
    return save_figure(fig, "population_log_likelihood")


def figure_population_transition_matrices(results_small, results_large, include_householder=True):
    """
    Report figure: the true A next to the A of the best fit for both numbers of cells per week and,
    if include_householder, the A reached from the Householder start. All fits have their states
    matched to MA, T, MU. Saves figures/population_transition_matrices.pdf, or
    figures/population_transition_matrices_best.pdf without the Householder panels.

    Parameters
    ----------
    results_small: dict from results/population_recovery_300.json
    results_large: dict from results/population_recovery_3000.json
    include_householder: bool, also show the A of the two Householder fits (default True)

    Returns
    -------
    pathlib.Path of the pdf written
    """
    apply_style()
    states = results_small["states"]
    small, large = results_small["cells_per_week"], results_large["cells_per_week"]
    panels = [
        (results_small["true"]["A"], "true $A$\n "),
        (results_small["fits"][results_small["best_index"]]["A"], f"best fit\n{small} cells/week"),
        (results_large["fits"][results_large["best_index"]]["A"], f"best fit\n{large} cells/week"),
    ]
    if include_householder:
        # The Householder start is always fits[0] (fit_from_several_starts puts it first)
        panels += [
            (results_small["fits"][0]["A"], f"Householder start\n{small} cells/week"),
            (results_large["fits"][0]["A"], f"Householder start\n{large} cells/week"),
        ]
    panel_width = 1.44   # inches per heatmap, the same in both versions of the figure
    fig, axes = plt.subplots(1, len(panels), figsize=(panel_width * len(panels), 1.75))
    for ax, (matrix, title) in zip(axes, panels):
        plot_matrix(matrix, states, states, ax, title)
        ax.set_xlabel("to")
    axes[0].set_ylabel("from")
    fig.subplots_adjust(wspace=0.35)
    if include_householder:
        return save_figure(fig, "population_transition_matrices")
    return save_figure(fig, "population_transition_matrices_best")


def figure_population_proportions(results_small, results_large):
    """
    Report figure: fitted p_t against the observed soft proportions, one panel per number of cells per week.
    Saves figures/population_proportions.pdf.

    Parameters
    ----------
    results_small: dict from results/population_recovery_300.json
    results_large: dict from results/population_recovery_3000.json

    Returns
    -------
    pathlib.Path of the pdf written
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2), sharey=True)
    for ax, results in zip(axes, [results_small, results_large]):
        plot_proportions(results, ax)
        ax.set_title(f"{results['cells_per_week']} cells per week, best fit", loc="left", fontsize=8)
    axes[1].set_ylabel("")
    fig.subplots_adjust(wspace=0.12)
    return save_figure(fig, "population_proportions")


def write_population_numbers_tex(results_small, results_large):
    """
    Write figures/population_numbers.tex: the numbers the report quotes about the toy population model,
    as LaTeX macros, so text and figures come from the same results files. "Small" in a macro name is
    the 300-cells-per-week run, "Large" the 3000-cells-per-week run. Use with \\input{figures/population_numbers}.

    Parameters
    ----------
    results_small: dict from results/population_recovery_300.json
    results_large: dict from results/population_recovery_3000.json

    Returns
    -------
    pathlib.Path of the file written
    """
    macros = {
        "popRandomStarts": results_small["number_of_random_starts"],
        "popMaxIterations": results_small["max_iterations"],
    }
    for size, results in [("Small", results_small), ("Large", results_large)]:
        fits = results["fits"]
        best = fits[results["best_index"]]
        householder = fits[0]   # fit_from_several_starts puts the Householder start first
        A_true = np.asarray(results["true"]["A"])
        A_best = np.asarray(best["A"])
        final_log_likelihoods = np.array([fit["log_likelihoods"][-1] for fit in fits])
        reaches_best = final_log_likelihoods > final_log_likelihoods.max() - 0.1   # ends within 0.1 of the best l

        macros[f"popCells{size}"] = results["cells_per_week"]
        macros[f"popLlTrue{size}"] = f"{results['true']['log_likelihood']:.2f}"
        macros[f"popLlBest{size}"] = f"{final_log_likelihoods.max():.2f}"
        macros[f"popLlHouseholder{size}"] = f"{householder['log_likelihoods'][-1]:.2f}"
        macros[f"popStartsAtBest{size}"] = int(reaches_best.sum())
        macros[f"popFirstStartAtBest{size}"] = int(np.argmax(reaches_best))
        macros[f"popBestStart{size}"] = results["best_index"]
        # Largest |fitted - true| in each row of A (rows MA, T, MU), and in pi and the means
        for j, state in enumerate(results["states"]):
            macros[f"popErr{state}{size}"] = f"{np.abs(A_best[j] - A_true[j]).max():.2f}"
        macros[f"popErrPi{size}"] = f"{np.abs(np.asarray(best['pi']) - results['true']['pi']).max():.2f}"
        macros[f"popErrMeans{size}"] = f"{np.abs(np.asarray(best['means']) - results['true']['means']).max():.2f}"
        macros[f"popErrHouseholder{size}"] = f"{np.abs(np.asarray(householder['A']) - A_true).max():.2f}"
        # Largest |fitted p_t - observed soft proportion| over all weeks and states
        macros[f"popProportionGap{size}"] = (
            f"{np.abs(np.asarray(results['best']['proportions']) - results['best']['soft_proportions']).max():.3f}")

    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / "population_numbers.tex"
    with open(path, "w") as f:
        f.write("% generated by plot_results.py from results/population_recovery_*.json; do not edit by hand\n")
        for name, value in macros.items():
            f.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")
    print("saved", path)
    return path


if __name__ == "__main__":
    results = load_results("baum_welch_restarts")
    figure_baum_welch(results)
    write_numbers_tex(results)

    results_small = load_results("population_recovery_300")
    results_large = load_results("population_recovery_3000")
    figure_population_cells(results_small)
    figure_population_log_likelihood(results_small, results_large)
    figure_population_transition_matrices(results_small, results_large)
    figure_population_transition_matrices(results_small, results_large, include_householder=False)
    figure_population_proportions(results_small, results_large)
    write_population_numbers_tex(results_small, results_large)
