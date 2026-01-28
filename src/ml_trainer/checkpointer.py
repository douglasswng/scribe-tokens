import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml_model.id import ModelId
from ml_trainer.config import TrainerConfig
from ml_trainer.state import TrainState
from utils.distributed_context import distributed_context


class Checkpointer:
    """Handles checkpoint saving, loading, and pruning."""

    def __init__(self, model_id: ModelId, config: TrainerConfig):
        self._model_id = model_id
        self._max_checkpoints = config.max_checkpoints

    @property
    def _best_checkpoint_symlink(self) -> Path:
        """Path to the symlink pointing to the best checkpoint."""
        return self._model_id.checkpoint_dir / "best.pt"

    def _get_state_dict(self, train_state: TrainState) -> dict[str, Any]:
        """Extract state dict from TrainState."""
        state_dict = {
            "model_state_dict": train_state.model.local_model.state_dict(),
            "optimiser_state_dict": train_state.optimiser.state_dict(),
            "scheduler_state_dict": train_state.scheduler.state_dict(),
            "epoch": train_state.epoch,
            "random_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
            "cudnn_state": {
                "deterministic": torch.backends.cudnn.deterministic,
                "benchmark": torch.backends.cudnn.benchmark,
            },
        }

        # Save CUDA random states if CUDA is available
        if torch.cuda.is_available():
            state_dict["random_state"]["torch_cuda"] = torch.cuda.get_rng_state_all()

        return state_dict

    def _load_state_dict(self, train_state: TrainState, state_dict: dict[str, Any]) -> TrainState:
        """Load state dict into TrainState."""
        train_state.model.local_model.load_state_dict(state_dict["model_state_dict"])
        train_state.optimiser.load_state_dict(state_dict["optimiser_state_dict"])
        train_state.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        train_state.epoch = state_dict["epoch"]

        # Restore random states if available (for backward compatibility)
        if "random_state" in state_dict:
            random_state = state_dict["random_state"]
            random.setstate(random_state["python"])
            np.random.set_state(random_state["numpy"])
            torch.set_rng_state(random_state["torch"])

            # Restore CUDA random states if they were saved
            if "torch_cuda" in random_state and random_state["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(random_state["torch_cuda"])

        # Restore cuDNN settings if available (for backward compatibility)
        if "cudnn_state" in state_dict:
            cudnn_state = state_dict["cudnn_state"]
            torch.backends.cudnn.deterministic = cudnn_state["deterministic"]
            torch.backends.cudnn.benchmark = cudnn_state["benchmark"]

        return train_state

    def _prune_checkpoints(self) -> None:
        """Keep the best checkpoint + N most recent checkpoints, removing all others."""
        if distributed_context.is_worker:
            return

        checkpoint_paths = self._model_id.list_checkpoint_paths()
        if len(checkpoint_paths) <= self._max_checkpoints:
            return

        # Resolve the best checkpoint symlink to get the actual checkpoint path
        best_checkpoint_path = None
        symlink = self._best_checkpoint_symlink
        if symlink.is_symlink():
            best_checkpoint_path = symlink.resolve()

        # Sort checkpoints by epoch number (oldest first)
        checkpoint_paths.sort(key=lambda p: int(p.stem.split("_")[1]))

        # Keep the N most recent checkpoints
        checkpoints_to_keep = set(checkpoint_paths[-self._max_checkpoints :])

        # Also keep the best checkpoint
        if best_checkpoint_path is not None:
            checkpoints_to_keep.add(best_checkpoint_path)

        # Remove all checkpoints that are not in the keep set
        for path in checkpoint_paths:
            if path.resolve() not in checkpoints_to_keep:
                path.unlink()

    def save_model(self, train_state: TrainState) -> None:
        """Save just the model weights."""
        if distributed_context.is_worker:
            return

        model_path = self._model_id.model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(train_state.model.local_model.state_dict(), model_path)

    def save_state(self, train_state: TrainState) -> None:
        """Save full training state to checkpoint."""
        if distributed_context.is_worker:
            return

        checkpoint_path = self._model_id.get_checkpoint_path(train_state.epoch)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._get_state_dict(train_state), checkpoint_path)
        self._prune_checkpoints()

    def mark_as_best(self, epoch: int) -> None:
        """Create/update symlink pointing to the best checkpoint."""
        if distributed_context.is_worker:
            return

        checkpoint_path = self._model_id.get_checkpoint_path(epoch)
        if not checkpoint_path.exists():
            return

        symlink = self._best_checkpoint_symlink
        if symlink.is_symlink():
            symlink.unlink()
        symlink.symlink_to(checkpoint_path.name)

    def load_state(self, epoch: int, train_state: TrainState) -> TrainState:
        """Load checkpoint for specific epoch."""
        checkpoint_path = self._model_id.get_checkpoint_path(epoch)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
        state_dict = torch.load(checkpoint_path, weights_only=False)
        return self._load_state_dict(train_state, state_dict)

    def load_best_state(self, train_state: TrainState) -> TrainState | None:
        """Load the best checkpoint if exists."""
        symlink = self._best_checkpoint_symlink
        if not symlink.exists():
            return None
        state_dict = torch.load(symlink, weights_only=False)
        return self._load_state_dict(train_state, state_dict)

    def load_latest_state(self, train_state: TrainState) -> TrainState | None:
        """Load most recent checkpoint if exists."""
        latest_path = self._model_id.get_latest_checkpoint_path()
        if latest_path is None:
            return None
        state_dict = torch.load(latest_path, weights_only=False)
        return self._load_state_dict(train_state, state_dict)
