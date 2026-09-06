"""Plotting helpers for multi-seed learning curves.

The house style for this repo: every curve is a mean over seeds with a shaded
band, never a single run. See `docs/experiment-protocol.md`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def aggregate(curves: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation across seeds, truncated to the shortest curve.

    Args:
        curves: one sequence of values per seed.

    Returns:
        `(mean, std)` arrays of length `min(len(c) for c in curves)`.
    """
    length = min(len(c) for c in curves)
    stacked = np.asarray([np.asarray(c[:length], dtype=float) for c in curves])
    return stacked.mean(axis=0), stacked.std(axis=0)


def smooth(values: Sequence[float], window: int = 10) -> np.ndarray:
    """Trailing moving average; returns the input unchanged if it is shorter than `window`."""
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_curves(
    series: Mapping[str, Sequence[Sequence[float]]],
    title: str,
    xlabel: str = "step",
    ylabel: str = "value",
    out_path: str | Path | None = None,
    smooth_window: int = 1,
):
    """Plot one mean±std band per label.

    Args:
        series: mapping from label to a list of per-seed curves.
        title: plot title.
        xlabel: x-axis label.
        ylabel: y-axis label.
        out_path: if given, save the figure there (parent dirs are created).
        smooth_window: trailing moving-average window applied before aggregating.

    Returns:
        The matplotlib `Figure`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, curves in series.items():
        prepared = [smooth(c, smooth_window) for c in curves] if smooth_window > 1 else list(curves)
        mean, std = aggregate(prepared)
        x = np.arange(mean.size)
        ax.plot(x, mean, label=label, linewidth=1.6)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, linewidth=0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
    return fig
