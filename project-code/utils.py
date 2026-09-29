"""
Shared helpers for the project. hmm.py holds the algorithms, experiments.py runs them,
plots.py draws figures; this file holds what they have in common.

Currently: saving and loading experiment results.
Every experiment saves one JSON file in results/<name>.json. That file is the single source
of truth: plots and the numbers quoted in the report are made from it, never from a live run.
To refresh a figure after changing values, rerun experiments.py (rewrites the JSON), then plots.py.
"""
import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"  # project-code/results/, created on first save


def _to_builtin(value):
    """
    Convert numpy arrays and scalars to plain Python (lists, float, int) so json can write them.
    Works recursively through dicts and lists, so a whole results dict can be passed in.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def save_results(name, results):
    """
    Save a results dict as results/<name>.json (overwrites an existing file).

    Parameters
    ----------
    name: str, file name without extension, e.g. "baum_welch_restarts"
    results: dict; may contain numpy arrays, numpy scalars, nested dicts and lists

    Returns
    -------
    pathlib.Path of the file written
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(_to_builtin(results), f, indent=2)
    return path


def load_results(name):
    """
    Load results/<name>.json. Raises FileNotFoundError if the experiment has not been run.

    Parameters
    ----------
    name: str, the name given to save_results

    Returns
    -------
    dict, the saved results; arrays come back as nested lists
    """
    with open(RESULTS_DIR / f"{name}.json") as f:
        return json.load(f)
