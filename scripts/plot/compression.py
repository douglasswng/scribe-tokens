"""Compression rate vs vocabulary size for each tokeniser type.

Loads results from the compression CSV and generates a publication-quality
PDF with one subplot per tokeniser type, showing compression rate across
vocabulary sizes and δ values.

Usage:
  python -m scripts.plot.compression
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from constants import FIGURES_DIR, RESULTS_DIR

RESULT_PATH = RESULTS_DIR / "compression.csv"
FIGURE_PATH = FIGURES_DIR / "compression.pdf"

OURS = "ScribeTokens"


def load_data(file_path: Path = RESULT_PATH) -> pd.DataFrame:
    return pd.read_csv(file_path)


def _setup_subplot_grid(n_tokenisers: int, figsize: tuple[float, float]) -> tuple[Figure, list]:
    """Set up the subplot grid based on number of tokenisers."""
    cols = 2
    rows = (n_tokenisers + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)

    # Always convert to a flat list for consistent indexing
    if isinstance(axes, np.ndarray):
        axes = axes.flatten().tolist()
    else:
        # Single subplot case
        axes = [axes]

    return fig, axes


def _plot_tokeniser_data(
    ax: Axes, tokeniser_data: pd.DataFrame, tokeniser_type: str, colors: list
) -> None:
    """Plot data for a single tokeniser type on the given axis."""
    deltas = sorted(tokeniser_data["delta"].unique(), reverse=True)

    for j, delta in enumerate(deltas):
        delta_data: pd.DataFrame = tokeniser_data[tokeniser_data["delta"] == delta].sort_values(
            "vocab_size"
        )  # type: ignore[call-overload]

        ax.plot(
            delta_data["vocab_size"] / 1000,
            delta_data["compression_rate"],
            color=colors[j % len(colors)],
            label=f"δ={delta}",
            marker="o",
            markersize=3,
            linewidth=1.5,
        )
    suffix = " (Ours)" if tokeniser_type == OURS else ""
    ax.set_title(f"{tokeniser_type}{suffix}", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=8)


def _finalize_plot_layout(fig: Figure, axes: list, n_tokenisers: int) -> None:
    """Hide unused subplots and add axis labels."""
    # Hide unused subplots
    for i in range(n_tokenisers, len(axes)):
        axes[i].set_visible(False)

    fig.supxlabel("Vocabulary Size (k)", fontsize=9)
    fig.supylabel("Compression Rate", fontsize=9)


def create_compression_plot(df: pd.DataFrame, figsize: tuple[float, float] = (5.5, 3.6)) -> Figure:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
        }
    )

    tokeniser_types = df["tokeniser_type"].unique()
    n_tokenisers = len(tokeniser_types)

    fig, axes = _setup_subplot_grid(n_tokenisers, figsize)
    fig.set_layout_engine("constrained")
    colors = plt.colormaps["viridis"](np.linspace(0, 1.0, 6))

    # Separate "ours" from other tokenisers
    tokeniser_types_list = list(tokeniser_types)
    ours_tokenisers = [t for t in tokeniser_types_list if t == OURS]
    other_tokenisers = [t for t in tokeniser_types_list if t != OURS]

    # Plot other tokenisers first, then ours at the end (bottom right)
    reordered_tokenisers = other_tokenisers + ours_tokenisers

    for i, tokeniser_type in enumerate(reordered_tokenisers):
        tokeniser_data: pd.DataFrame = df[df["tokeniser_type"] == tokeniser_type]  # type: ignore[assignment]
        _plot_tokeniser_data(axes[i], tokeniser_data, tokeniser_type, list(colors))

    _finalize_plot_layout(fig, axes, n_tokenisers)

    # Shared legend from last subplot's handles
    handles, labels = axes[n_tokenisers - 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", ncols=1, fontsize=8)

    return fig


def save_figure(fig: Figure, output_path: Path = FIGURE_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)


def plot(
    input_path: Path = RESULT_PATH,
    output_path: Path = FIGURE_PATH,
    figsize: tuple[float, float] = (5.5, 3.6),
    show: bool = False,
) -> None:
    df = load_data(input_path)
    fig = create_compression_plot(df, figsize)
    save_figure(fig, output_path)

    if show:
        plt.show()


if __name__ == "__main__":
    plot()
