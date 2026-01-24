from pathlib import Path
from typing import Any

import torch

from ml_model.id import ModelId
from ml_trainer.config import TrainerConfig
from ml_trainer.state import TrainState


class Checkpointer:
    """Handles checkpoint saving, loading, and pruning."""

    def __init__(self, model_id: ModelId, config: TrainerConfig):
        self._model_id = model_id
        self._checkpoint_interval = config.checkpoint_interval
        self._max_checkpoints = config.max_checkpoints

    @property
    def _best_checkpoint_symlink(self) -> Path:
        """Path to the symlink pointing to the best checkpoint."""
        return self._model_id.checkpoint_dir / "best.pt"

    def _get_state_dict(self, train_state: TrainState) -> dict[str, Any]:
        """Extract state dict from TrainState."""
        return {
            "model_state_dict": train_state.model.local_model.state_dict(),
            "optimiser_state_dict": train_state.optimiser.state_dict(),
            "scheduler_state_dict": train_state.scheduler.state_dict(),
            "epoch": train_state.epoch,
        }

    def _load_state_dict(self, train_state: TrainState, state_dict: dict[str, Any]) -> TrainState:
        """Load state dict into TrainState."""
        train_state.model.local_model.load_state_dict(state_dict["model_state_dict"])
        train_state.optimiser.load_state_dict(state_dict["optimiser_state_dict"])
        train_state.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        train_state.epoch = state_dict["epoch"]
        return train_state

    def _prune_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        checkpoint_paths = self._model_id.list_checkpoint_paths()
        if len(checkpoint_paths) <= self._max_checkpoints:
            return

        checkpoint_paths.sort(key=lambda p: int(p.stem.split("_")[1]))
        for path in checkpoint_paths[: -self._max_checkpoints]:
            path.unlink()

    def save_model(self, train_state: TrainState) -> None:
        """Save just the model weights."""
        model_path = self._model_id.model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(train_state.model.local_model.state_dict(), model_path)

    def save_state(self, train_state: TrainState) -> None:
        """Save full training state to checkpoint."""
        checkpoint_path = self._model_id.get_checkpoint_path(train_state.epoch)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._get_state_dict(train_state), checkpoint_path)
        self._prune_checkpoints()

    def mark_as_best(self, epoch: int) -> None:
        """Create/update symlink pointing to the best checkpoint."""
        checkpoint_path = self._model_id.get_checkpoint_path(epoch)
        if not checkpoint_path.exists():
            return

        symlink = self._best_checkpoint_symlink
        if symlink.is_symlink():
            symlink.unlink()
        symlink.symlink_to(checkpoint_path.name)

    def should_save_state(self, train_state: TrainState) -> bool:
        """Check if checkpoint should be saved based on interval."""
        return (train_state.epoch - 1) % self._checkpoint_interval == 0

    def load_state(self, epoch: int, train_state: TrainState) -> TrainState:
        """Load checkpoint for specific epoch."""
        checkpoint_path = self._model_id.get_checkpoint_path(epoch)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
        state_dict = torch.load(checkpoint_path, weights_only=True)
        return self._load_state_dict(train_state, state_dict)

    def load_best_state(self, train_state: TrainState) -> TrainState | None:
        """Load the best checkpoint if exists."""
        symlink = self._best_checkpoint_symlink
        if not symlink.exists():
            return None
        state_dict = torch.load(symlink, weights_only=True)
        return self._load_state_dict(train_state, state_dict)

    def load_latest_state(self, train_state: TrainState) -> TrainState | None:
        """Load most recent checkpoint if exists."""
        latest_path = self._model_id.get_latest_checkpoint_path()
        if latest_path is None:
            return None
        state_dict = torch.load(latest_path, weights_only=True)
        return self._load_state_dict(train_state, state_dict)
