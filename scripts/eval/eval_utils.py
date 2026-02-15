"""Utility functions for evaluation scripts."""

import csv
from pathlib import Path
from typing import Iterator

from torch.utils.data import DataLoader

import constants
from dataloader.create import create_dataloaders
from ml_model.id import ModelId, Task


def get_test_dataloader(model_id: ModelId) -> DataLoader:
    """Get the test dataloader for a given model."""
    _, _, test_dataloader = create_dataloaders(model_id)
    return test_dataloader


def get_model_ids_for_tasks(tasks: set[Task]) -> Iterator[ModelId]:
    """Iterate over all model IDs for the given tasks."""
    for model_id in ModelId.create_defaults():
        if model_id.task in tasks:
            yield model_id


def compute_cer(pred_text: str, target_text: str) -> float:
    """Compute Character Error Rate using Levenshtein distance.

    CER = edit_distance(pred, target) / len(target)

    Args:
        pred_text: Predicted text string
        target_text: Ground truth text string

    Returns:
        Character Error Rate as a float
    """
    if len(target_text) == 0:
        return 1.0 if len(pred_text) > 0 else 0.0

    m, n = len(pred_text), len(target_text)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_text[i - 1] == target_text[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / len(target_text)


def should_skip_evaluation(results_path: Path, model_id: ModelId) -> bool:
    """Check if evaluation should be skipped for a model.

    Returns True if:
    - The model file doesn't exist
    - The result already exists in the CSV

    Args:
        results_path: Path to the CSV file
        model_id: Model identifier

    Returns:
        True if evaluation should be skipped
    """
    # Check if model exists
    if not model_id.model_path.exists():
        print(f"Skipping {model_id}: model not found at {model_id.model_path}")
        return True

    # Check if result already exists
    if results_path.exists():
        with open(results_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for existing_row in reader:
                if (
                    len(existing_row) >= 2
                    and existing_row[0] == str(model_id)
                    and existing_row[1] == constants.DATASET
                ):
                    print(f"Skipping {model_id} on {constants.DATASET}: already evaluated")
                    return True

    return False


def add_metrics(results_path: Path, model_id: ModelId, metrics: dict[str, float]) -> None:
    """Add metrics to a CSV results file.

    Note: Call should_skip_evaluation() before this to avoid duplicates.

    Args:
        results_path: Path to the CSV file
        model_id: Model identifier
        metrics: Dictionary of metric names to values
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["model_id", "dataset"] + list(metrics.keys())
    row = [str(model_id), constants.DATASET] + list(metrics.values())

    if results_path.exists():
        with open(results_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    else:
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(row)

    print(f"Added {model_id} on {constants.DATASET}: {metrics}")
