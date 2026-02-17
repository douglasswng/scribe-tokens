"""
Convert evaluation CSV results to LaTeX tables.

Usage:
  python -m scripts.plot.results
  python -m scripts.plot.results --task HTR
"""

import argparse
from pathlib import Path

import pandas as pd

from constants import RESULTS_DIR, TABLES_DIR

# Define which methods are "ours" (will be marked in table)
OUR_METHODS = {"ScribeTokens"}

# Define metric properties: (higher_is_better, format_spec, display_name)
METRIC_CONFIG = {
    "cer": (False, ".2f", "CER (\\%) $\\downarrow$"),
    "accuracy": (True, ".2f", "Acc (\\%) $\\uparrow$"),
}

# Multiplier for display (e.g., 0.107 -> 10.7%)
METRIC_SCALE = {
    "cer": 100,
    "accuracy": 100,
}


def parse_model_id(model_id: str) -> tuple[str, str]:
    """Extract task and representation from model_id string.

    Example: "Task: HTR, Repr: ScribeTokens-8 (vocab_size: 32000)"
    Returns: ("HTR", "ScribeTokens-8")
    """
    parts = model_id.split(", Repr: ")
    task = parts[0].replace("Task: ", "")
    repr_full = parts[1] if len(parts) > 1 else ""
    # Extract just the representation name (before parentheses)
    repr_name = repr_full.split(" (")[0] if " (" in repr_full else repr_full
    return task, repr_name


def format_value(value: float, metric: str, is_best: bool) -> str:
    """Format a metric value, bolding if best."""
    scale = METRIC_SCALE.get(metric, 1)
    fmt = METRIC_CONFIG[metric][1]
    formatted = f"{value * scale:{fmt}}"
    if is_best:
        return f"\\textbf{{{formatted}}}"
    return formatted


def format_delta(delta: float, metric: str) -> str:
    """Format a delta value with color: green for improvement, red for degradation."""
    scale = METRIC_SCALE.get(metric, 1)
    fmt = METRIC_CONFIG[metric][1]
    higher_is_better = METRIC_CONFIG[metric][0]

    scaled = delta * scale
    sign = "+" if scaled >= 0 else ""
    formatted = f"{sign}{scaled:{fmt}}"

    is_improvement = (higher_is_better and delta > 0) or (not higher_is_better and delta < 0)
    color = "teal" if is_improvement else "red"

    return f"{{\\scriptsize\\textcolor{{{color}}}{{({formatted})}}}}"


def is_our_method(repr_name: str) -> bool:
    """Check if this representation is one of our methods."""
    return any(ours in repr_name for ours in OUR_METHODS)


def strip_token_suffix(repr_name: str) -> str:
    """Remove -8 suffix from token-based representations (not Point-5)."""
    if repr_name.endswith("-8"):
        return repr_name[:-2]
    return repr_name


def format_repr_name(repr_name: str) -> str:
    """Format representation name for display, marking ours."""
    display = strip_token_suffix(repr_name)
    if is_our_method(repr_name):
        return f"{display} (Ours)"
    return display


DATASET_DISPLAY = {
    "deepwriting": "DeepWriting",
    "iam": "IAM",
}

DATASET_ORDER = ["iam", "deepwriting"]

TRAINING_TYPES = ["", "+PT"]


def csv_to_latex(csv_path: Path, output_path: Path) -> str:
    """Convert a results CSV to a LaTeX table with datasets as column groups
    and Base/+SFT as sub-rows per method."""
    df = pd.read_csv(csv_path)

    # Parse model_id into task and representation
    parsed = df["model_id"].apply(parse_model_id)
    df["task"] = parsed.apply(lambda x: x[0])
    df["repr"] = parsed.apply(lambda x: x[1])

    # Determine base task name and training type
    df["training"] = df["task"].apply(lambda t: "+PT" if "_SFT" in t else "")
    base_task = df["task"].apply(lambda t: t.replace("_SFT", "")).iloc[0]

    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    reprs = df["repr"].unique()
    metrics = [col for col in df.columns if col in METRIC_CONFIG]

    # Build LaTeX tabular
    lines = []

    # Column spec: method + training + (metrics per dataset)
    num_data_cols = len(datasets) * len(metrics)
    col_spec = "ll" + "c" * num_data_cols
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row 1: Dataset names spanning metrics
    header1_parts = ["", ""]
    for dataset in datasets:
        display = DATASET_DISPLAY.get(dataset, dataset)
        header1_parts.append(f"\\multicolumn{{{len(metrics)}}}{{c}}{{{display}}}")
    lines.append(" & ".join(header1_parts) + " \\\\")

    # cmidrules for each dataset group
    cmidrules = []
    col_idx = 3  # Start after method and training columns
    for _ in datasets:
        end_col = col_idx + len(metrics) - 1
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{end_col}}}")
        col_idx = end_col + 1
    lines.append(" ".join(cmidrules))

    # Header row 2: Metric names
    header2_parts = ["Method", ""]
    for _ in datasets:
        for metric in metrics:
            header2_parts.append(METRIC_CONFIG[metric][2])
    lines.append(" & ".join(header2_parts) + " \\\\")
    lines.append("\\midrule")

    # Find best values per column (dataset, metric) across all rows
    best_values: dict[tuple[str, str], float] = {}
    for dataset in datasets:
        subset = df[df["dataset"] == dataset]
        for metric in metrics:
            higher_is_better = METRIC_CONFIG[metric][0]
            if higher_is_better:
                best_val = subset[metric].max()
            else:
                best_val = subset[metric].min()
            best_values[(dataset, metric)] = float(best_val)

    # Precompute baseline (w/o PT) values for delta calculation
    baseline_values: dict[tuple[str, str, str], float] = {}
    for repr_name in reprs:
        for dataset in datasets:
            base_df = df[
                (df["task"] == base_task) & (df["repr"] == repr_name) & (df["dataset"] == dataset)
            ]
            assert isinstance(base_df, pd.DataFrame)
            if len(base_df) > 0:
                for metric in metrics:
                    baseline_values[(repr_name, dataset, metric)] = float(base_df[metric].iloc[0])

    # Sort so our methods appear last
    repr_order = sorted(reprs, key=lambda r: (1 if is_our_method(r) else 0, r))

    for i, repr_name in enumerate(repr_order):
        for j, training in enumerate(TRAINING_TYPES):
            row_parts = []

            # Method name only on first sub-row
            if j == 0:
                row_parts.append(format_repr_name(repr_name))
            else:
                row_parts.append("")

            row_parts.append(training)

            task_name = base_task if training == "" else f"{base_task}_SFT"

            for dataset in datasets:
                row_df = df[
                    (df["task"] == task_name)
                    & (df["repr"] == repr_name)
                    & (df["dataset"] == dataset)
                ]
                assert isinstance(row_df, pd.DataFrame)

                for metric in metrics:
                    if len(row_df) == 0:
                        row_parts.append("--")
                    else:
                        value = float(row_df[metric].iloc[0])
                        best = best_values[(dataset, metric)]
                        is_best = abs(value - best) < 1e-9
                        cell = format_value(value, metric, is_best)

                        # Append delta on w/ PT rows
                        if training == "+PT":
                            bkey = (repr_name, dataset, metric)
                            if bkey in baseline_values:
                                delta = value - baseline_values[bkey]
                                cell += " " + format_delta(delta, metric)

                        row_parts.append(cell)

            lines.append(" & ".join(row_parts) + " \\\\")

        # Visual separation between method groups
        if i + 1 < len(repr_order):
            if not is_our_method(repr_name) and is_our_method(repr_order[i + 1]):
                lines.append("\\midrule")
            elif not is_our_method(repr_name):
                lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    latex_content = "\n".join(lines)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_content)
    print(f"Wrote LaTeX table to {output_path}")

    return latex_content


def main(task: str | None = None) -> None:
    """Generate LaTeX tables from CSV results."""
    TASK_CSVS = {"htr", "htg"}
    if task:
        csv_files = [RESULTS_DIR / f"{task.lower()}.csv"]
    else:
        csv_files = [f for f in RESULTS_DIR.glob("*.csv") if f.stem in TASK_CSVS]

    for csv_path in csv_files:
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}")
            continue

        output_path = TABLES_DIR / f"{csv_path.stem}.tex"
        latex = csv_to_latex(csv_path, output_path)
        print(f"\n{latex}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CSV results to LaTeX tables")
    parser.add_argument("--task", type=str, help="Specific task (e.g., HTR, HTG)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(task=args.task)
