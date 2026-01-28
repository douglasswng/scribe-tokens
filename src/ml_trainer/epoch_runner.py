import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ml_trainer.batch_processor import BatchProcessor
from ml_trainer.checkpointer import Checkpointer
from ml_trainer.config import TrainerConfig
from ml_trainer.early_stopper import EarlyStopper
from ml_trainer.progress import ProgressFactory
from ml_trainer.state import TrainState
from ml_trainer.stats import TrainStats
from ml_trainer.tracker import Tracker
from utils.distributed_context import distributed_context


class EpochRunner:
    """Orchestrates training and validation epochs."""

    def __init__(
        self,
        batch_processor: BatchProcessor,
        tracker: Tracker,
        config: TrainerConfig,
    ):
        self._batch_processor = batch_processor
        self._tracker = tracker
        self._config = config

    def prepare_epoch(
        self,
        train_state: TrainState,
        train_stats: TrainStats,
        train_loader: DataLoader,
    ) -> None:
        """Prepare for a new epoch."""
        train_state.epoch += 1
        train_stats.new_epoch(train_state.epoch)

        if distributed_context.is_distributed:
            if not isinstance(train_loader.sampler, DistributedSampler):
                raise ValueError("Expected DistributedSampler in distributed training")
            train_loader.sampler.set_epoch(train_state.epoch)

    def run_training_epoch(
        self,
        train_state: TrainState,
        train_stats: TrainStats,
        train_loader: DataLoader,
        num_epochs: int,
    ) -> None:
        """Run one training epoch."""
        train_state.model.train()
        pbar = ProgressFactory.create(
            train_loader, f"Epoch {train_state.epoch}/{num_epochs} (Train)"
        )

        for batch in pbar:
            batch = self._batch_processor.move_batch_to_device(batch)
            self._batch_processor.process_train_batch(train_state, train_stats, batch)
            if distributed_context.is_master:
                pbar.set_postfix(train_stats.curr_train_batch_stats.summary_dict)

        if distributed_context.is_master:
            self._tracker.log_metrics(train_stats.curr_train_epoch_stats.full_dict)

    def run_validation_epoch(
        self,
        train_state: TrainState,
        train_stats: TrainStats,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> None:
        """Run one validation epoch."""
        train_state.model.eval()
        pbar = ProgressFactory.create(val_loader, f"Epoch {train_state.epoch}/{num_epochs} (Val)")

        with torch.no_grad():
            monitor_batch_idx = self._select_monitor_batch(pbar, train_state.epoch)

            for batch_idx, batch in enumerate(pbar):
                batch = self._batch_processor.move_batch_to_device(batch)
                if batch_idx == monitor_batch_idx:
                    train_state.model.monitor(batch)

                self._batch_processor.process_validation_batch(train_state, train_stats, batch)
                if distributed_context.is_master:
                    pbar.set_postfix(train_stats.curr_val_batch_stats.summary_dict)

        if distributed_context.is_master:
            self._tracker.log_metrics(train_stats.curr_val_epoch_stats.full_dict)

    def complete_epoch(
        self,
        train_state: TrainState,
        train_stats: TrainStats,
        early_stopper: EarlyStopper,
        checkpointer: Checkpointer,
    ) -> bool:
        """Complete epoch, check early stopping, save checkpoints. Returns should_stop."""
        early_stopper.register_stats(train_stats.curr_val_epoch_stats)
        is_best = early_stopper.is_best

        if distributed_context.is_master:
            checkpointer.save_state(train_state)
            if is_best:
                checkpointer.mark_as_best(train_state.epoch)

        if distributed_context.is_master:
            tqdm.write(f"Patience Counter: {early_stopper.counter}/{early_stopper.patience}")
            tqdm.write(train_stats.curr_epoch_stats_formatted)

        return early_stopper.should_stop

    def _select_monitor_batch(self, pbar: tqdm, epoch: int) -> int | None:
        """Select a random batch for monitoring (master only)."""
        if not distributed_context.is_master:
            return None
        if len(pbar) == 0:
            return None
        return random.randint(0, len(pbar) - 1)
