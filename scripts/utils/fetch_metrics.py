"""Fetch val_epoch metrics from all SwanLab experiments and save to CSV.

For every dataset (iam, deepwriting), for every run, for every time step,
records the val_epoch_* metrics.

Usage:
  python -m scripts.utils.fetch_metrics
"""

import pandas as pd
import swanlab
from swanlab.api.experiment import Experiment

from constants import RESULTS_DIR

USERNAME = "douglasswng"
DATASETS = ["iam", "deepwriting"]
OUTPUT_PATH = RESULTS_DIR / "metrics.csv"


def get_val_epoch_keys(experiment: Experiment) -> list[str]:
    """Get val_epoch_* metric keys via the columns API."""
    resp, _ = experiment._client.get(f"/experiment/{experiment.id}/column")  # type: ignore[index]
    return [col["key"] for col in resp["list"] if col["key"].startswith("val_epoch_")]  # type: ignore[index]


def fetch_all_metrics() -> pd.DataFrame:
    api = swanlab.Api()
    all_rows: list[pd.DataFrame] = []

    for dataset in DATASETS:
        project = f"scribe-tokens-{dataset}"
        path = f"{USERNAME}/{project}"
        print(f"Fetching {project}...")

        for run in api.runs(path):
            keys = get_val_epoch_keys(run)
            if not keys:
                print(f"  Skipping {run.name} (no val_epoch metrics)")
                continue

            df = run.metrics(keys=keys)
            # Drop timestamp columns
            df = df[[c for c in df.columns if not c.endswith("_timestamp")]]
            df = df.dropna(how="all")

            df.insert(0, "dataset", dataset)
            df.insert(1, "run", run.name)
            df.index.name = "epoch"

            all_rows.append(df)
            print(f"  {run.name}: {len(df)} epochs, keys={keys}")

    return pd.concat(all_rows, ignore_index=False)


def main() -> None:
    df = fetch_all_metrics()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH)
    print(f"\nSaved {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
