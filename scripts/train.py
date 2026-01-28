"""
Unified training script for ScribeTokens models.

Supports:
- Single model training: python -m scripts.train --task HTR --repr scribe
- All models sequential: python -m scripts.train --all
- Quick test mode: python -m scripts.train --all --test
- Distributed training: torchrun --nproc_per_node=2 -m scripts.train --all --test
"""

import argparse

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset

from constants import GRAD_ACCUM_STEPS, LEARNING_RATE, TRACKERS_DIR, WEIGHT_DECAY
from dataloader.create import _create_dataloader, create_dataloaders
from dataloader.dataset import create_datasets
from dataloader.split import create_datasplit
from ink_repr.id import VectorReprId, VectorType
from ink_tokeniser.id import TokeniserId, TokenType
from ml_model.factory import ModelFactory
from ml_model.id import ModelId, Task
from ml_trainer.checkpointer import Checkpointer
from ml_trainer.config import TrainerConfig
from ml_trainer.state import TrainState
from ml_trainer.tracker import SwanLabTracker, Tracker
from ml_trainer.trainer import Trainer
from scripts.train_utils import get_params_dict
from utils.clear_folder import clear_folder
from utils.distributed_context import distributed_context
from utils.set_random_seed import set_random_seed


def load_train_state(model_id: ModelId, config: TrainerConfig) -> TrainState:
    """Load or create training state for the given model."""
    model = ModelFactory.create(model_id)
    print(f"Loaded model of size {float(model.num_params) / 1e6:.2f}M params")
    optimiser = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=torch.cuda.is_available(),  # better performance
    )
    scheduler = LambdaLR(optimiser, lr_lambda=lambda _: 1.0)
    train_state = TrainState(model, optimiser, scheduler)

    latest_train_state = Checkpointer(model_id, config).load_latest_state(train_state)
    if latest_train_state is not None:
        print(f"Found latest checkpoint for {model_id}, resuming training")
        train_state = latest_train_state
    else:
        print(f"No latest checkpoint found for {model_id}, starting new training")
    return train_state


def setup_tracker(model_id: ModelId, experiment_name: str, test_mode: bool = False) -> Tracker:
    """Set up experiment tracking for the training run."""
    tracker = SwanLabTracker()
    tracker.begin_experiment(experiment_name, TRACKERS_DIR)

    tags = [model_id.task, str(model_id.repr_id)]
    run_name = str(model_id)

    if test_mode:
        tags.append("test")
        run_name = f"test_{model_id}"

    tracker.begin_run(tags=tags, run_name=run_name)
    tracker.log_params(get_params_dict())
    return tracker


def train_single_model(
    model_id: ModelId,
    experiment_name: str,
    test_mode: bool = False,
    num_epochs: int | None = None,
    batch_size: int | None = None,
) -> None:
    """Train a single model with optional test mode."""

    if test_mode:
        # Quick test with small dataset
        batch_size = batch_size or 1
        num_epochs = num_epochs or 2

        datasplit = create_datasplit()
        train_dataset, val_dataset, _ = create_datasets(model_id.repr_id, datasplit)

        # Divide by accumulation steps to get per-step batch size
        per_step_batch_size = batch_size // GRAD_ACCUM_STEPS

        # Subset to 1.5 batches (using per-step batch size)
        subset_size = int(1.5 * per_step_batch_size)
        if len(train_dataset) > subset_size:
            train_dataset = Subset(train_dataset, list(range(subset_size)))
        if len(val_dataset) > subset_size:
            val_dataset = Subset(val_dataset, list(range(subset_size)))

        train_loader = _create_dataloader(
            train_dataset, batch_size=per_step_batch_size, shuffle=True
        )
        val_loader = _create_dataloader(val_dataset, batch_size=per_step_batch_size, shuffle=False)
    else:
        # Normal training
        train_loader, val_loader, _ = create_dataloaders(model_id)
        num_epochs = num_epochs or model_id.task.num_epochs

    config = TrainerConfig(patience=model_id.task.patience, grad_accum_steps=GRAD_ACCUM_STEPS)
    train_state = load_train_state(model_id, config)
    trainer = Trainer(model_id, setup_tracker(model_id, experiment_name, test_mode), config)
    trainer.train(train_state, train_loader, val_loader, num_epochs)
    distributed_context.barrier()


def parse_repr_id(repr_str: str):
    """Parse representation ID from command-line string.

    Args:
        repr_str: String representation, e.g., 'scribe', 'point5', 'rel', 'text'

    Returns:
        ReprId: Either a TokeniserId or VectorReprId
    """
    match repr_str.lower():
        case "point5":
            return VectorReprId(VectorType.POINT5)
        case "scribe":
            return TokeniserId.create_scribe()
        case "rel":
            return TokeniserId.create_rel()
        case "text":
            return TokeniserId.create_text()
        case _:
            raise ValueError(
                f"Unknown representation: {repr_str}. Valid options: point5, scribe, rel, text"
            )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified training script for ScribeTokens models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model training
  python -m scripts.train --task HTR --repr scribe

  # Train all default models (sequential)
  python -m scripts.train --all

  # Quick test with small dataset
  python -m scripts.train --all --test

  # Distributed training (2 GPUs)
  torchrun --nproc_per_node=2 -m scripts.train --all --test

  # Custom experiment name
  python -m scripts.train --all --experiment-name "MyExperiment"
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Train all default model combinations",
    )
    mode_group.add_argument(
        "--task",
        type=str,
        choices=[task.value for task in Task],
        help="Training task to perform (requires --repr)",
    )

    # Single model options
    parser.add_argument(
        "--repr",
        type=str,
        help="Representation ID (e.g., scribe, point5, rel, text, point3, abs)",
    )

    # Shared options
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test mode with small dataset (1.5 batches, 2 epochs)",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment name (default: ScribeTokens0128 or ScribeTokensTest)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (mainly for test mode)",
    )

    args = parser.parse_args()

    # Validation
    if args.task and not args.repr:
        parser.error("--task requires --repr")
    if args.repr and not args.task:
        parser.error("--repr requires --task")

    return args


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Determine experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    elif args.test:
        experiment_name = "ScribeTokensTest"
    else:
        experiment_name = "ScribeTokens0202"

    # Clear trackers on master process if in test mode
    if args.test and distributed_context.is_master:
        clear_folder(TRACKERS_DIR, confirm=False)

    distributed_context.barrier()

    if args.all:
        # Train all default models
        model_ids = ModelId.create_defaults()

        for model_id in model_ids:
            # Skip non-scribe tokenisers in test mode
            if args.test:
                if (
                    isinstance(tokeniser_id := model_id.repr_id, TokeniserId)
                    and tokeniser_id.type != TokenType.SCRIBE
                ):
                    print(f"Test mode: skipping {model_id}")
                    continue

            # Check if already trained
            if model_id.model_path.exists():
                print(f"Model {model_id} already trained at {model_id.model_path}")
                if not args.test:
                    print("Skipping. Delete the model file to retrain.")
                    continue

            print(f"\nTraining model: {model_id}")
            train_single_model(
                model_id,
                experiment_name,
                test_mode=args.test,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
            )
            print(f"✓ Training complete: {model_id}")

    else:
        # Train single model
        task = Task(args.task)
        repr_id = parse_repr_id(args.repr)
        model_id = ModelId(task=task, repr_id=repr_id)

        print(f"Training model: {model_id}")

        if model_id.model_path.exists() and not args.test:
            print(f"Model {model_id} already trained at {model_id.model_path}")
            print("Skipping training. Delete the model file to retrain.")
            return

        train_single_model(
            model_id,
            experiment_name,
            test_mode=args.test,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(f"✓ Training complete! Model saved to {model_id.model_path}")


if __name__ == "__main__":
    set_random_seed(42)
    main()
