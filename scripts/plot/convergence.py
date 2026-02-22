"""Generate LaTeX convergence-speedup table from training metrics.

For HTR and HTG, shows per-method and per-dataset:
  - Convergence epoch without pretraining (base)
  - Convergence epoch with pretraining (+PT)
  - Epochs for +PT to reach the base model's best val loss
  - Speedup factor (base convergence / epochs-to-match)

Usage:
  python -m scripts.plot.convergence
"""

from typing import cast

import pandas as pd

from constants import RESULTS_DIR, TABLES_DIR

METRICS_PATH = RESULTS_DIR / "metrics.csv"
OUTPUT_DIR = TABLES_DIR

OUR_METHODS = {"ScribeTokens"}
DATASET_ORDER = ["iam", "deepwriting"]
DATASET_DISPLAY = {"iam": "IAM", "deepwriting": "DeepWriting"}
TASKS = ["HTR", "HTG"]


def parse_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the raw metrics CSV into a clean frame with task/repr/val_loss."""
    df = df.copy()
    df["val_loss"] = df["val_epoch_ce"].fillna(df["val_epoch_nll"])
    parsed = df["run"].str.extract(r"Task: (\w+), Repr: (.+)")
    df["task"] = parsed[0]
    df["repr"] = parsed[1].str.split(r" \(").str[0]
    return df


def convergence_epoch(series: pd.DataFrame) -> int:
    """Return the epoch at which val_loss is minimised."""
    return int(series.loc[series["val_loss"].idxmin(), "epoch"]) + 1


def epochs_to_match(series: pd.DataFrame, target: float) -> int | None:
    """Return the first epoch where val_loss <= target, or None if never reached."""
    matched = series[series["val_loss"] <= target]
    if matched.empty:
        return None
    return int(matched["epoch"].min()) + 1


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary table with convergence stats for each task/repr/dataset."""
    rows = []
    for task in TASKS:
        for dataset in DATASET_ORDER:
            for repr_name in sorted(df["repr"].unique()):
                base = cast(
                    pd.DataFrame,
                    df[
                        (df["task"] == task)
                        & (df["repr"] == repr_name)
                        & (df["dataset"] == dataset)
                    ],
                )
                sft = cast(
                    pd.DataFrame,
                    df[
                        (df["task"] == f"{task}_SFT")
                        & (df["repr"] == repr_name)
                        & (df["dataset"] == dataset)
                    ],
                )

                if base.empty or sft.empty:
                    continue

                base_conv = convergence_epoch(base)
                base_best = float(base["val_loss"].min())
                match_epoch = epochs_to_match(sft, base_best)
                if match_epoch is not None:
                    speedup = base_conv / match_epoch
                else:
                    speedup = None

                # Look up NTP pretraining convergence epoch for this repr/dataset
                ntp = cast(
                    pd.DataFrame,
                    df[
                        (df["task"] == "NTP")
                        & (df["repr"] == repr_name)
                        & (df["dataset"] == dataset)
                    ],
                )
                pt_epochs = convergence_epoch(ntp) if not ntp.empty else None

                rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "repr": repr_name,
                        "pt_epochs": pt_epochs,
                        "base_conv": base_conv,
                        "match_epoch": match_epoch,
                        "speedup": speedup,
                    }
                )

    return pd.DataFrame(rows)


def is_our_method(repr_name: str) -> bool:
    return any(ours in repr_name for ours in OUR_METHODS)


def strip_token_suffix(repr_name: str) -> str:
    if repr_name.endswith("-8"):
        return repr_name[:-2]
    return repr_name


def format_repr_name(repr_name: str) -> str:
    display = strip_token_suffix(repr_name)
    if is_our_method(repr_name):
        return f"{display} (Ours)"
    return display


def _format_dataset_cells(r, best_speedup: float | None) -> list[str]:
    """Format the 4 data cells (PT Ep., No PT, +PT, Spd.) for one dataset."""
    parts = []
    parts.append(str(int(r["pt_epochs"])) if pd.notna(r["pt_epochs"]) else "{--}")
    parts.append(str(int(r["base_conv"])))

    if pd.notna(r["match_epoch"]):
        parts.append(str(int(r["match_epoch"])))
    else:
        parts.append("{--}")

    if pd.notna(r["speedup"]):
        speedup_str = f"{r['speedup']:.1f} $\\times$"
        if best_speedup is not None and abs(r["speedup"] - best_speedup) < 1e-9:
            speedup_str = f"\\bfseries {speedup_str}"
        parts.append(speedup_str)
    else:
        parts.append("{--}")

    return parts


def _compute_col_specs(task_summary: pd.DataFrame) -> list[str]:
    """Compute siunitx S column format strings for each dataset's 4 columns."""
    specs = []
    for dataset in DATASET_ORDER:
        subset = cast(pd.DataFrame, task_summary[task_summary["dataset"] == dataset])

        pt_vals = [int(v) for v in subset["pt_epochs"].dropna()]
        max_val = max(pt_vals) if pt_vals else 99
        specs.append(f"S[table-format={len(str(max_val))}]")

        base_vals = [int(v) for v in subset["base_conv"].dropna()]
        max_val = max(base_vals) if base_vals else 99
        specs.append(f"S[table-format={len(str(max_val))}]")

        match_vals = [int(v) for v in subset["match_epoch"].dropna()]
        max_val = max(match_vals) if match_vals else 9
        specs.append(f"S[table-format={len(str(max_val))}]")

        spd_vals = list(subset["speedup"].dropna())
        int_digits = len(str(int(max(spd_vals)))) if spd_vals else 1
        specs.append(f"S[table-format={int_digits}.1, table-space-text-post=$\\times$]")

    return specs


def to_latex(task_summary: pd.DataFrame) -> str:
    """Render a task slice as a LaTeX booktabs table with both datasets."""
    # Determine repr ordering from the union across datasets
    all_reprs = set()
    for dataset in DATASET_ORDER:
        subset = cast(pd.DataFrame, task_summary[task_summary["dataset"] == dataset])
        all_reprs.update(subset["repr"].unique())
    reprs = sorted(all_reprs, key=lambda r: (1 if is_our_method(r) else 0, r))

    # Best speedup per dataset for bolding
    best_speedup = {}
    for dataset in DATASET_ORDER:
        subset = cast(pd.DataFrame, task_summary[task_summary["dataset"] == dataset])
        valid = subset["speedup"].dropna()
        best_speedup[dataset] = valid.max() if not valid.empty else None

    col_specs = _compute_col_specs(task_summary)
    indent = "  "
    col_spec_str = "l\n" + "\n".join(
        indent + " ".join(col_specs[i : i + 4]) for i in range(0, len(col_specs), 4)
    )
    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{col_spec_str}}}")
    lines.append("\\toprule")

    # Multicolumn dataset header
    mc_parts = [""]
    for idx, dataset in enumerate(DATASET_ORDER):
        mc_parts.append(f"\\multicolumn{{4}}{{c}}{{{DATASET_DISPLAY[dataset]}}}")
    lines.append(" & ".join(mc_parts) + " \\\\")

    # cmidrules
    n_datasets = len(DATASET_ORDER)
    rules = []
    for idx in range(n_datasets):
        start = 2 + idx * 4
        end = start + 3
        rules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(rules))

    # Sub-header — wrap column names in {} for siunitx S columns
    sub = ["Method"]
    for _ in DATASET_ORDER:
        sub.extend(["{PT Ep.}", "{No PT}", "{+PT}", "{Spd.}"])
    lines.append(" & ".join(sub) + " \\\\")
    lines.append("\\midrule")

    for i, repr_name in enumerate(reprs):
        parts = [format_repr_name(repr_name)]
        for dataset in DATASET_ORDER:
            subset = cast(pd.DataFrame, task_summary[task_summary["dataset"] == dataset])
            row = cast(pd.DataFrame, subset[subset["repr"] == repr_name])
            if row.empty:
                parts.extend(["{--}"] * 4)
            else:
                parts.extend(_format_dataset_cells(row.iloc[0], best_speedup[dataset]))

        lines.append(" & ".join(parts) + " \\\\")

        if i + 1 < len(reprs):
            if not is_our_method(repr_name) and is_our_method(reprs[i + 1]):
                lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(METRICS_PATH)
    df = parse_metrics(df)
    summary = build_table(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(summary.to_string(index=False))
    print()

    for task in TASKS:
        task_summary = cast(pd.DataFrame, summary[summary["task"] == task])
        if task_summary.empty:
            continue

        latex = to_latex(task_summary)
        output_path = OUTPUT_DIR / f"conv_{task.lower()}.tex"
        output_path.write_text(latex)
        print(f"Wrote {output_path}")
        print(latex)
        print()


if __name__ == "__main__":
    main()
