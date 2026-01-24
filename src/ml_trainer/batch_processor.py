import torch
import torch.distributed as dist
from torch import Tensor

from ml_trainer.distributed_ops import DistributedOps
from ml_trainer.gradient_handler import GradientHandler
from ml_trainer.stats import BatchStats, TrainStats
from ml_trainer.state import TrainState
from ml_trainer.tracker import Tracker
from schemas.batch import Batch
from utils.distributed_context import distributed_context


class BatchProcessor:
    """Processes individual training and validation batches."""

    def __init__(self, gradient_handler: GradientHandler, tracker: Tracker):
        self._gradient_handler = gradient_handler
        self._tracker = tracker

    def process_train_batch(
        self,
        train_state: TrainState,
        train_stats: TrainStats,
        batch: Batch,
    ) -> None:
        """Process a single training batch."""
        losses: dict[str, Tensor] = train_state.model(batch)
        total_loss = sum(losses.values(), torch.tensor(0.0))

        train_state.optimiser.zero_grad()
        max_grad = self._gradient_handler.compute_gradients(train_state.model, total_loss)
        self._gradient_handler.clip_and_step(train_state)

        if distributed_context.is_distributed:
            DistributedOps.reduce_losses(losses)
            dist.all_reduce(max_grad, op=dist.ReduceOp.MAX)

        batch_stats = BatchStats(
            losses={k: v.item() for k, v in losses.items()},
            max_grad=max_grad.item(),
            lr=float(train_state.scheduler.get_last_lr()[0]),
        )
        train_stats.add_train_batch_stats(batch_stats)

        if distributed_context.is_master:
            self._tracker.log_metrics(batch_stats.full_dict)

    def process_validation_batch(
        self,
        train_state: TrainState,
        train_stats: TrainStats,
        batch: Batch,
    ) -> None:
        """Process a single validation batch."""
        losses = train_state.model(batch)

        if distributed_context.is_distributed:
            DistributedOps.reduce_losses(losses)

        batch_stats = BatchStats(losses={k: v.item() for k, v in losses.items()})
        train_stats.add_val_batch_stats(batch_stats)
