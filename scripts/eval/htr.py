"""Evaluation script for HTR (Handwriting Text Recognition) models."""

from torch.utils.data import DataLoader

from constants import RESULTS_DIR
from ml_model.factory import ModelFactory
from ml_model.id import ModelId, Task
from ml_model.locals.htr import HTRModel
from schemas.batch import Batch
from scripts.eval.eval_utils import (
    add_metrics,
    compute_cer,
    get_model_ids_for_tasks,
    get_test_dataloader,
    should_skip_evaluation,
)

RESULTS_PATH = RESULTS_DIR / "htr.csv"
TASKS = {Task.HTR, Task.HTR_SFT}


def calculate_batch_metrics(model: HTRModel, batch: Batch) -> dict[str, list[float]]:
    """Calculate HTR metrics for a single batch."""
    instances = batch.instances
    pred_texts = model.batch_predict_text(instances)
    true_texts = [instance.parsed.text for instance in instances]

    cers = [
        compute_cer(pred_text, true_text) for pred_text, true_text in zip(pred_texts, true_texts)
    ]
    accuracies = [
        float(pred_text == true_text) for pred_text, true_text in zip(pred_texts, true_texts)
    ]

    return {"cer": cers, "accuracy": accuracies}


def calculate_metrics(model: HTRModel, dataloader: DataLoader) -> dict[str, float]:
    """Calculate aggregate HTR metrics over the entire dataloader."""
    all_metrics: dict[str, list[float]] = {"cer": [], "accuracy": []}

    for batch in dataloader:
        batch = batch.to_device()
        batch_metrics = calculate_batch_metrics(model, batch)
        all_metrics["cer"].extend(batch_metrics["cer"])
        all_metrics["accuracy"].extend(batch_metrics["accuracy"])

    return {
        "cer": sum(all_metrics["cer"]) / len(all_metrics["cer"]),
        "accuracy": sum(all_metrics["accuracy"]) / len(all_metrics["accuracy"]),
    }


def evaluate_model(model_id: ModelId) -> None:
    """Evaluate a single HTR model."""
    if should_skip_evaluation(RESULTS_PATH, model_id):
        return

    model = ModelFactory.load_pretrained(model_id)
    model.eval()
    assert isinstance(model, HTRModel)

    dataloader = get_test_dataloader(model_id)
    metrics = calculate_metrics(model, dataloader)
    add_metrics(RESULTS_PATH, model_id, metrics)


def main() -> None:
    """Evaluate all HTR models."""
    for model_id in get_model_ids_for_tasks(TASKS):
        evaluate_model(model_id)


if __name__ == "__main__":
    main()
