"""Plot validation loss curves for IAM HTR (no SFT) token-based methods.

Loads cross-entropy loss values from the metrics CSV and generates a
publication-quality PDF showing the training dynamics for each tokenisation.

Usage:
  python -m scripts.plot.double_descent
"""

from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from constants import FIGURES_DIR, RESULTS_DIR

METRICS_PATH = RESULTS_DIR / "metrics.csv"
OUTPUT_PATH = FIGURES_DIR / "double_descent.pdf"

# Display names and colours for each representation
REPR_STYLE: dict[str, tuple[str, str]] = {
    "RelTokens-8": ("RelTokens", "#1f77b4"),
    "TextTokens-8": ("TextTokens", "#ff7f0e"),
    "ScribeTokens-8": ("ScribeTokens (Ours)", "#d62728"),
}


def main() -> None:
    df = pd.read_csv(METRICS_PATH)

    # IAM, HTR, no SFT, token-based methods only (cross-entropy loss)
    mask = (
        (df["dataset"] == "iam")
        & df["run"].str.contains("Task: HTR,")
        & ~df["run"].str.contains("SFT")
        & ~df["run"].str.contains("Point-5")
        & df["val_epoch_ce"].notna()
    )
    df = cast(pd.DataFrame, df[mask]).copy()

    # Extract short repr name (e.g. "ScribeTokens-8")
    df["repr"] = cast(pd.Series, df["run"]).str.extract(r"Repr: (.+?) \(")[0]

    # --- Plot setup (match compression.pdf style) ---
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(3.4, 2.1), layout="constrained")

    for repr_name, (label, colour) in REPR_STYLE.items():
        subset = cast(pd.DataFrame, df[df["repr"] == repr_name]).sort_values("epoch")
        if subset.empty:
            continue
        ax.plot(
            subset["epoch"] + 1,
            subset["val_epoch_ce"],
            label=label,
            color=colour,
            linewidth=1.5,
        )

    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Validation Cross-Entropy Loss", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
