import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_model.id import ModelId
from ml_trainer.batch_processor import BatchProcessor
from ml_trainer.checkpointer import Checkpointer
from ml_trainer.config import TrainerConfig
from ml_trainer.early_stopper import EarlyStopper
from ml_trainer.epoch_runner import EpochRunner
from ml_trainer.gradient_handler import GradientHandler
from ml_trainer.progress import ProgressFactory
from ml_trainer.state import TrainState
from ml_trainer.stats import TrainStats
from ml_trainer.tracker import Tracker
from utils.distributed_context import distributed_context


class Trainer:
    """Main training orchestrator using composition."""

    def __init__(self, model_id: ModelId, tracker: Tracker, config: TrainerConfig):
        self._tracker = tracker
        self._config = config
        self._configure_backends()

        self._checkpointer = Checkpointer(model_id, config)
        self._early_stopper = EarlyStopper(patience=config.patience)
        self._gradient_handler = GradientHandler(config.max_grad_norm)
        self._batch_processor = BatchProcessor(self._gradient_handler, tracker, use_amp=True)
        self._epoch_runner = EpochRunner(self._batch_processor, tracker, config)

    def _configure_backends(self) -> None:
        """Configure PyTorch backends for optimal performance."""
        if not torch.cuda.is_available():
            return

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def train(
        self,
        train_state: TrainState,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> None:
        """Execute the full training loop."""
        train_stats = self._initialise_training(train_state)
        epoch_pbar = ProgressFactory.create(range(train_state.epoch, num_epochs), "Epochs")

        for _ in epoch_pbar:
            self._epoch_runner.prepare_epoch(train_state, train_stats, train_loader)
            self._epoch_runner.run_training_epoch(
                train_state, train_stats, train_loader, num_epochs
            )
            self._epoch_runner.run_validation_epoch(
                train_state, train_stats, val_loader, num_epochs
            )

            should_stop = self._epoch_runner.complete_epoch(
                train_state, train_stats, self._early_stopper, self._checkpointer
            )
            if should_stop:
                break

        self._finalise_training(train_state, train_stats)

    def _initialise_training(self, train_state: TrainState) -> TrainStats:
        """Initialize training components."""
        train_state.model.set_tracker(self._tracker)
        return TrainStats()

    def _finalise_training(self, train_state: TrainState, train_stats: TrainStats) -> None:
        """Finalize training, save best model."""
        if distributed_context.is_worker:
            return

        best_state = self._checkpointer.load_best_state(train_state) or train_state
        self._checkpointer.save_model(best_state)
        self._tracker.end_run()

        tqdm.write(f"Training completed in {train_stats.time_ellapsed_formatted}")
