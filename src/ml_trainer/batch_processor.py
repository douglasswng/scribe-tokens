import torch
import torch.distributed as dist
from torch import Tensor
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from ml_trainer.distributed_ops import DistributedOps
from ml_trainer.gradient_handler import GradientHandler
from ml_trainer.state import TrainState
from ml_trainer.stats import BatchStats, TrainStats
from ml_trainer.tracker import Tracker
from schemas.batch import Batch
from schemas.instance import Instance
from utils.distributed_context import distributed_context


class BatchProcessor:
    """Processes individual training and validation batches with mixed precision."""

    def __init__(self, gradient_handler: GradientHandler, tracker: Tracker, use_amp: bool = True):
        self._gradient_handler = gradient_handler
        self._tracker = tracker
        self._use_amp = use_amp and torch.cuda.is_available()
        self._scaler = GradScaler("cuda") if self._use_amp else None

    def move_batch_to_device(self, batch: Batch) -> Batch:
        """Move all tensors in the batch to the appropriate device."""
        device = distributed_context.device
        moved_instances = [
            Instance(
                parsed=instance.parsed,
                repr_id=instance.repr_id,
                repr=instance.repr.to(device, non_blocking=True),
                char=instance.char.to(device, non_blocking=True),
            )
            for instance in batch.instances
        ]
        return Batch(instances=moved_instances)

    def process_train_batch(
        self, train_state: TrainState, train_stats: TrainStats, batch: Batch
    ) -> None:
        """Process a single training batch."""
        losses = self._forward_pass(train_state.model, batch)
        total_loss = sum(losses.values(), torch.tensor(0.0, device=distributed_context.device))

        max_grad = self._backward_pass(train_state, total_loss)
        self._sync_distributed(losses, max_grad)
        self._update_stats(train_stats, losses, max_grad, train_state)

    def process_validation_batch(
        self, train_state: TrainState, train_stats: TrainStats, batch: Batch
    ) -> None:
        """Process a single validation batch."""
        losses = self._forward_pass(train_state.model, batch)
        self._sync_distributed(losses)
        train_stats.add_val_batch_stats(BatchStats(losses={k: v.item() for k, v in losses.items()}))

    def _forward_pass(self, model: torch.nn.Module, batch: Batch) -> dict[str, Tensor]:
        """Run forward pass with optional mixed precision."""
        if self._use_amp:
            with autocast("cuda", dtype=torch.bfloat16):
                return model(batch)
        return model(batch)

    def _backward_pass(self, train_state: TrainState, total_loss: Tensor) -> Tensor:
        """Run backward pass with optional gradient scaling."""
        train_state.optimiser.zero_grad()

        if self._scaler is not None:
            self._scaler.scale(total_loss).backward()
            self._scaler.unscale_(train_state.optimiser)
        else:
            total_loss.backward()

        max_grad = self._gradient_handler.get_max_grad(train_state.model)
        self._gradient_handler.clip(train_state)
        train_state.optimiser.step()
        train_state.scheduler.step()

        if self._scaler is not None:
            self._scaler.update()

        return max_grad

    def _sync_distributed(self, losses: dict[str, Tensor], max_grad: Tensor | None = None) -> None:
        """Synchronize losses and gradients across distributed processes."""
        if not distributed_context.is_distributed:
            return

        DistributedOps.reduce_losses(losses)
        if max_grad is not None:
            dist.all_reduce(max_grad, op=dist.ReduceOp.MAX)

    def _update_stats(
        self,
        train_stats: TrainStats,
        losses: dict[str, Tensor],
        max_grad: Tensor,
        train_state: TrainState,
    ) -> None:
        """Update training statistics and log metrics."""
        batch_stats = BatchStats(
            losses={k: v.item() for k, v in losses.items()},
            max_grad=max_grad.item(),
            lr=float(train_state.scheduler.get_last_lr()[0]),
        )
        train_stats.add_train_batch_stats(batch_stats)

        if distributed_context.is_master:
            self._tracker.log_metrics(batch_stats.full_dict)
