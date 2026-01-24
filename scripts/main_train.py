from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from constants import (
    BATCH_SIZE,
    DELTA,
    DROPOUT,
    FFN_FACTOR,
    HIDDEN_DIM,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_HEADS,
    NUM_LAYERS,
    PATIENCE,
    TRACKERS_DIR,
    VOCAB_SIZE,
    WEIGHT_DECAY,
)
from dataloader.create import create_dataloaders
from ml_model.factory import ModelFactory
from ml_model.id import ModelId
from ml_trainer.checkpointer import Checkpointer
from ml_trainer.state import TrainState
from ml_trainer.tracker import SwanLabTracker, Tracker
from ml_trainer.trainer import Trainer

EXPERIMENT_NAME = "ScribeTokens0122"


def load_train_state(model_id: ModelId) -> TrainState:
    model = ModelFactory.create(model_id)
    print(f"Loaded model of size {float(model.num_params) / 1e6:.2f}M params")
    optimiser = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = LambdaLR(optimiser, lr_lambda=lambda epoch: 1.0)
    train_state = TrainState(model, optimiser, scheduler)

    latest_train_state = Checkpointer(model_id).load_latest_state(train_state)
    if latest_train_state is not None:
        print(f"Found latest checkpoint for {model_id}, resuming training")
        train_state = latest_train_state
    else:
        print(f"No latest checkpoint found for {model_id}, starting new training")
    return train_state


def setup_tracker(model_id: ModelId) -> Tracker:
    tracker = SwanLabTracker()
    tracker.begin_experiment(EXPERIMENT_NAME, TRACKERS_DIR)
    tracker.begin_run(tags=[model_id.task, str(model_id.repr_id)], run_name=str(model_id))
    tracker.log_params(
        {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "delta": DELTA,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "dropout": DROPOUT,
            "vocab_size": VOCAB_SIZE,
            "ffn_factor": FFN_FACTOR,
        }
    )
    return tracker


def train_with_resume(model_id: ModelId) -> None:
    train_loader, val_loader, _ = create_dataloaders(model_id)
    train_state = load_train_state(model_id)
    trainer = Trainer(model_id, setup_tracker(model_id), PATIENCE)
    trainer.train(train_state, train_loader, val_loader, NUM_EPOCHS)
    distributed_context.barrier()  # sometimes training breaks without barrier I think?


def main() -> None:
    model_ids = ModelId.create_defaults()
    for model_id in model_ids:
        if model_id.model_path.exists():
            print(f"Model {model_id} already trained")
            continue

        train_with_resume(model_id)


if __name__ == "__main__":
    from constants import CHECKPOINTS_DIR, TRACKERS_DIR
    from utils.clear_folder import clear_folder
    from utils.distributed_context import distributed_context
    from utils.set_random_seed import set_random_seed

    if distributed_context.is_master:
        clear_folder(CHECKPOINTS_DIR, confirm=True)
        clear_folder(TRACKERS_DIR, confirm=False)
        # clear_folder(MODELS_DIR, confirm=True)

    distributed_context.barrier()
    set_random_seed(42)
    main()
