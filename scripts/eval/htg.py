"""Evaluation script for HTG (Handwriting Text Generation) models.

Uses the scribe HTR_SFT model to recognize generated inks and compute CER/accuracy.
"""

from torch.utils.data import DataLoader

from constants import RESULTS_DIR
from ink_tokeniser.id import TokeniserId
from ml_model.factory import ModelFactory
from ml_model.id import ModelId, Task
from ml_model.locals.htg import HTGModel
from ml_model.locals.htr import HTRModel
from schemas.batch import Batch
from schemas.instance import Instance
from schemas.parsed import Parsed
from scripts.eval.eval_utils import (
    add_metrics,
    compute_cer,
    get_model_ids_for_tasks,
    get_test_dataloader,
    should_skip_evaluation,
)

RESULTS_PATH = RESULTS_DIR / "htg.csv"
TASKS = {Task.HTG, Task.HTG_SFT}


def load_htr_model() -> HTRModel:
    """Load the scribe HTR_SFT model for evaluating generated inks."""
    htr_model_id = ModelId(task=Task.HTR_SFT, repr_id=TokeniserId.create_scribe())  # best HTR model
    model = ModelFactory.load_pretrained(htr_model_id)
    model.eval()
    assert isinstance(model, HTRModel)
    return model


def calculate_batch_metrics(
    htg_model: HTGModel, htr_model: HTRModel, batch: Batch
) -> dict[str, list[float]]:
    """Calculate HTG metrics for a single batch using HTR model for recognition."""
    instances = batch.instances
    true_texts = [instance.parsed.text for instance in instances]

    # Generate inks from text using HTG model
    pred_inks_batched = htg_model.batch_generate_inks(instances, num_generations=1)
    pred_inks = [inks[0] for inks in pred_inks_batched]

    # Create instances from generated inks for HTR recognition (with empty text)
    scribe_repr_id = TokeniserId.create_scribe()
    htr_instances = [
        Instance.from_parsed(Parsed(ink=ink, text=""), scribe_repr_id).to_device()
        for ink in pred_inks
    ]

    # Recognize text from generated inks
    pred_texts = htr_model.batch_predict_text(htr_instances)

    cers = [
        compute_cer(pred_text, true_text) for pred_text, true_text in zip(pred_texts, true_texts)
    ]
    accuracies = [
        float(pred_text == true_text) for pred_text, true_text in zip(pred_texts, true_texts)
    ]

    return {"cer": cers, "accuracy": accuracies}


def calculate_metrics(
    htg_model: HTGModel, htr_model: HTRModel, dataloader: DataLoader
) -> dict[str, float]:
    """Calculate aggregate HTG metrics over the entire dataloader."""
    all_metrics: dict[str, list[float]] = {"cer": [], "accuracy": []}

    for batch in dataloader:
        batch = batch.to_device()
        batch_metrics = calculate_batch_metrics(htg_model, htr_model, batch)
        all_metrics["cer"].extend(batch_metrics["cer"])
        all_metrics["accuracy"].extend(batch_metrics["accuracy"])

    return {
        "cer": sum(all_metrics["cer"]) / len(all_metrics["cer"]),
        "accuracy": sum(all_metrics["accuracy"]) / len(all_metrics["accuracy"]),
    }


def evaluate_model(model_id: ModelId, htr_model: HTRModel) -> None:
    """Evaluate a single HTG model."""
    if should_skip_evaluation(RESULTS_PATH, model_id):
        return

    model = ModelFactory.load_pretrained(model_id)
    model.eval()
    assert isinstance(model, HTGModel)

    dataloader = get_test_dataloader(model_id)
    metrics = calculate_metrics(model, htr_model, dataloader)
    add_metrics(RESULTS_PATH, model_id, metrics)


def main() -> None:
    """Evaluate all HTG models."""
    htr_model = load_htr_model()
    for model_id in get_model_ids_for_tasks(TASKS):
        evaluate_model(model_id, htr_model)


if __name__ == "__main__":
    main()
