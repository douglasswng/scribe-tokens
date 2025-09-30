from typing import Iterable, Sized, Tuple
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from core.train.stats import TrainStats, BatchStats
from core.train.state import TrainState
from core.train.checkpointer import Checkpointer
from core.train.early_stopper import EarlyStopper
from core.model.tracker import Tracker
from core.model.paths import ModelPaths
from core.data_schema.batch import Batch
from core.utils.distributed_context import distributed_context


def create_progress_bar(iterable: Iterable, desc: str) -> tqdm:
    total = len(iterable) if isinstance(iterable, Sized) else None
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=False,
        disable=distributed_context.is_worker,
    )


def reduce_tensor_across_processes(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor across all processes using average reduction."""
    if not distributed_context.is_distributed:
        return tensor

    if distributed_context.backend == "nccl":
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    elif distributed_context.backend == "gloo":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / distributed_context.world_size
    else:
        raise ValueError(f"Unsupported backend: {distributed_context.backend}")
    
    return tensor


def compute_gradients_and_get_max(model: torch.nn.Module, loss: torch.Tensor) -> torch.Tensor:
    """Compute gradients from loss and return the maximum gradient norm for monitoring."""
    loss.backward()
    
    # Compute maximum gradient for monitoring
    max_grad = torch.tensor(0.0, device=distributed_context.device)
    for param in model.parameters():
        if param.grad is not None:
            max_grad = torch.max(max_grad, param.grad.data.abs().max())
    
    return max_grad


def clip_gradients_and_update(train_state: TrainState, max_grad_norm: float) -> None:
    """Clip gradients and apply optimizer and scheduler updates."""
    clip_grad_norm_(train_state.model.parameters(), max_norm=max_grad_norm)
    train_state.optimiser.step()
    train_state.scheduler.step()


def process_train_batch(train_state: TrainState, train_stats: TrainStats, batch: Batch, 
                       tracker: Tracker, max_grad_norm: float) -> None:
    """Process a single training batch."""
    losses = train_state.model(batch)
    total_loss = sum(losses.values())
    assert isinstance(total_loss, torch.Tensor)
    
    train_state.optimiser.zero_grad()
    max_grad = compute_gradients_and_get_max(train_state.model, total_loss)
    clip_gradients_and_update(train_state, max_grad_norm)

    if distributed_context.is_distributed:
        for loss_tensor in losses.values():
            reduce_tensor_across_processes(loss_tensor)
        dist.all_reduce(max_grad, op=dist.ReduceOp.MAX)

    batch_stats = BatchStats(
        losses={k: v.item() for k, v in losses.items()},
        max_grad=max_grad.item(),
        lr=train_state.scheduler.get_last_lr()[0]
    )
    train_stats.add_train_batch_stats(batch_stats)

    if distributed_context.is_master:
        tracker.log_metrics(batch_stats.full_dict)


def process_validation_batch(train_state: TrainState, train_stats: TrainStats, batch: Batch) -> None:
    """Process a single validation batch."""
    losses = train_state.model(batch)
    
    # Reduce losses across processes
    if distributed_context.is_distributed:
        for loss_tensor in losses.values():
            reduce_tensor_across_processes(loss_tensor)
    
    batch_stats = BatchStats(losses={k: v.item() for k, v in losses.items()})
    train_stats.add_val_batch_stats(batch_stats)


def prepare_epoch(train_state: TrainState, train_stats: TrainStats, train_loader: DataLoader) -> None:
    """Prepare for a new epoch by updating state and creating progress bars."""
    train_state.epoch += 1
    train_stats.new_epoch(train_state.epoch)

    # Set epoch for distributed sampler
    if distributed_context.is_distributed:
        if not isinstance(train_loader.sampler, DistributedSampler):
            raise ValueError("Expected DistributedSampler when using distributed training")
        train_loader.sampler.set_epoch(train_state.epoch)


def complete_epoch(train_state: TrainState, train_stats: TrainStats, 
                  early_stopper: EarlyStopper, checkpointer: Checkpointer) -> bool:
    """Complete the epoch by checking early stopping and saving checkpoints."""
    should_stop = early_stopper.should_stop(train_stats.curr_val_epoch_stats)
    should_save_state = (
        early_stopper.should_save_state or 
        checkpointer.should_save_state(train_state)
    )
    
    if should_save_state and distributed_context.is_master:
        checkpointer.save_state(train_state)
    
    if distributed_context.is_master:
        tqdm.write(f"Patience Counter: {early_stopper.counter}/{early_stopper.patience}")
        tqdm.write(train_stats.curr_epoch_stats_formatted)
    
    return should_stop


def initialise_training(train_state: TrainState, num_epochs: int) -> Tuple[TrainStats, tqdm]:
    """Initialise training by creating stats tracker and progress bar."""
    train_stats = TrainStats()
    epoch_pbar = create_progress_bar(range(train_state.epoch, num_epochs), "Epochs")
    return train_stats, epoch_pbar


def finalise_training(train_state: TrainState, train_stats: TrainStats,
                     early_stopper: EarlyStopper, checkpointer: Checkpointer, tracker: Tracker) -> None:
    """Finalise training by saving best model and ending tracker run."""
    if distributed_context.is_worker:
        return

    best_epoch = early_stopper.best_epoch
    best_state = checkpointer.load_state(best_epoch, train_state)
    checkpointer.save_model(best_state)
    tracker.end_run()

    tqdm.write(f"Training completed in {train_stats.time_ellapsed_formatted}")


class Trainer:
    def __init__(self,
                 model_paths: ModelPaths,
                 tracker: Tracker,
                 max_grad_norm: float=1.0):
        self._tracker = tracker
        self._max_grad_norm = max_grad_norm

        self._checkpointer = Checkpointer(model_paths)
        self._early_stopper = EarlyStopper()  

    def _run_training_epoch(self, train_state: TrainState, train_stats: TrainStats,
                            train_loader: DataLoader, num_epochs: int) -> None:
        """Run one training epoch."""
        train_state.model.train()
        train_pbar = create_progress_bar(train_loader,
                                       f"Epoch {train_state.epoch}/{num_epochs} (Train)")
        for batch in train_pbar:
            process_train_batch(train_state, train_stats, batch, self._tracker, self._max_grad_norm)
            if distributed_context.is_master:
                train_pbar.set_postfix(train_stats.curr_train_batch_stats.summary_dict)

        if distributed_context.is_master:
            self._tracker.log_metrics(train_stats.curr_train_epoch_stats.full_dict)

    def _run_validation_epoch(self, train_state: TrainState, train_stats: TrainStats,
                              val_loader: DataLoader, num_epochs: int) -> None:
        """Run one validation epoch."""
        train_state.model.eval()
        val_pbar = create_progress_bar(val_loader,
                                     f"Epoch {train_state.epoch}/{num_epochs} (Val)")
        with torch.no_grad():
            # Randomly select a batch for monitoring (only on master process)
            monitor_batch_idx = None
            if distributed_context.is_master and len(val_pbar) > 0:
                monitor_batch_idx = random.randint(0, len(val_pbar) - 1)
            
            for batch_idx, batch in enumerate(val_pbar):
                if batch_idx == monitor_batch_idx:
                    train_state.model.monitor(batch)

                process_validation_batch(train_state, train_stats, batch)
                if distributed_context.is_master:
                    val_pbar.set_postfix(train_stats.curr_val_batch_stats.summary_dict)

        if distributed_context.is_master:
            self._tracker.log_metrics(train_stats.curr_val_epoch_stats.full_dict)

    def _execute_training_loop(self, train_state: TrainState, train_stats: TrainStats,
                              train_loader: DataLoader, val_loader: DataLoader,
                              epoch_pbar: tqdm, num_epochs: int) -> None:
        """Execute the main training loop."""
        for _ in epoch_pbar:
            prepare_epoch(train_state, train_stats, train_loader)
            self._run_training_epoch(train_state, train_stats, train_loader, num_epochs)
            self._run_validation_epoch(train_state, train_stats, val_loader, num_epochs)
            
            should_stop = complete_epoch(train_state, train_stats, self._early_stopper, self._checkpointer)
            if should_stop:
                break

    def train(self, train_state: TrainState, train_loader: DataLoader, 
              val_loader: DataLoader, num_epochs: int):
        train_stats, epoch_pbar = initialise_training(train_state, num_epochs)
        self._execute_training_loop(train_state, train_stats,
                                    train_loader, val_loader,
                                    epoch_pbar, num_epochs)
        finalise_training(train_state, train_stats, self._early_stopper, self._checkpointer, self._tracker)