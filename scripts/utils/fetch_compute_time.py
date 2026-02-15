"""Fetch and print total compute time from all SwanLab experiments.

Usage:
  python -m scripts.utils.fetch_compute_time
"""

import swanlab

USERNAME = "douglasswng"
DATASETS = ["iam", "deepwriting"]


def main() -> None:
    api = swanlab.Api()
    total_s = 0.0

    for dataset in DATASETS:
        project = f"scribe-tokens-{dataset}"
        path = f"{USERNAME}/{project}"
        print(f"\n{project}")

        for run in api.runs(path):
            duration_ms = run._data.get("duration")
            if duration_ms is None:
                continue

            duration_s = duration_ms / 1000
            total_s += duration_s
            hours, rem = divmod(duration_s, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"  {run.name}: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    total_h, rem = divmod(total_s, 3600)
    total_m, total_sec = divmod(rem, 60)
    print(f"\nTotal compute time: {int(total_h):02d}:{int(total_m):02d}:{int(total_sec):02d}")


if __name__ == "__main__":
    main()
