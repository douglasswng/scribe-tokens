#!/usr/bin/env python3
"""Generate discretization artifacts figure for the paper.

Shows how quantization parameter δ affects digital ink reconstruction quality,
comparing raw quantized output (left column) with postprocessed/smoothed output
(right column) for each δ value.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from schemas.ink import DigitalInk, Point, Stroke

SAMPLE_PATH = "data/iam/parsed/a01-007z-07.json"
OUTPUT_PATH = Path("output/figures/discretization.pdf")


def load_ink_from_json(path: str) -> DigitalInk:
    """Load an IAM ink JSON file into a DigitalInk object."""
    with open(path) as f:
        data = json.load(f)
    strokes = []
    for s in data["ink"]["strokes"]:
        points = [Point(x=pt["x"], y=pt["y"]) for pt in s["points"]]
        strokes.append(Stroke(points=points))
    return DigitalInk(strokes=strokes)


def quantize_and_rescale(ink: DigitalInk, delta: float) -> DigitalInk:
    """Quantize ink: scale by 1/δ, round, scale back by δ."""
    return ink.scale(1 / delta).discretise().scale(delta)


def quantize_and_postprocess(ink: DigitalInk, delta: float) -> DigitalInk:
    """Quantize ink and apply the tokeniser's postprocessor (smooth)."""
    result = quantize_and_rescale(ink, delta)
    result = result.downsample(factor=1)
    result = result.smooth(window_length=7, polyorder=3)
    return result


def get_bounding_box(ink: DigitalInk) -> tuple[float, float, float, float]:
    """Compute the bounding box (xmin, xmax, ymin, ymax)."""
    all_x, all_y = [], []
    for stroke in ink.strokes:
        for pt in stroke.points:
            all_x.append(pt.x)
            all_y.append(pt.y)
    return min(all_x), max(all_x), min(all_y), max(all_y)


def plot_ink(ax: Axes, ink: DigitalInk):
    """Plot ink strokes on an axes."""
    for stroke in ink.strokes:
        xs = [pt.x for pt in stroke.points]
        ys = [pt.y for pt in stroke.points]
        if len(xs) == 1:
            ax.scatter(xs, ys, s=10, c="k", zorder=2)
        else:
            ax.plot(xs, ys, "-k", linewidth=1.0)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ink = load_ink_from_json(SAMPLE_PATH)

    deltas = [1, 2, 4, 8, 16, 32, 64, 128]
    nrows = len(deltas)
    ncols = 2  # left: raw quantized, right: postprocessed

    # --- Matplotlib rcParams for publication style ---
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
            "font.size": 10,
        }
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7, 7),
        gridspec_kw={"hspace": 0.05, "wspace": 0.02},
    )

    # Bounding box from the original ink for consistent limits
    xmin, xmax, ymin, ymax = get_bounding_box(ink)
    x_pad = (xmax - xmin) * 0.05
    y_pad = (ymax - ymin) * 0.05

    # Column titles
    axes[0, 0].set_title("Quantized", fontsize=10, fontweight="bold")
    axes[0, 1].set_title("Quantized + Smoothed", fontsize=10, fontweight="bold")

    for row_idx, delta in enumerate(deltas):
        # Left column: raw quantized
        raw = quantize_and_rescale(ink, delta)
        # Right column: quantized + postprocessed
        smooth = quantize_and_postprocess(ink, delta)

        for col_idx, result_ink in enumerate([raw, smooth]):
            ax = axes[row_idx, col_idx]
            plot_ink(ax, result_ink)

            # Consistent coordinate limits
            ax.set_xlim(xmin - x_pad, xmax + x_pad)
            ax.set_ylim(ymin - y_pad, ymax + y_pad)
            ax.invert_yaxis()
            ax.set_aspect("equal")

            # Remove all ticks, labels, spines
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Row label (δ value) on the left subplot
        label = f"$\\delta = {delta}$"
        if delta == 8:
            axes[row_idx, 0].set_ylabel(
                label,
                fontsize=10,
                fontweight="bold",
                color="#2e7d32",
                rotation=0,
                labelpad=30,
                va="center",
            )
        else:
            axes[row_idx, 0].set_ylabel(
                label,
                fontsize=10,
                rotation=0,
                labelpad=30,
                va="center",
            )

    fig.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=300)
    print(f"Saved {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
