from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ml_model.model import Model


@dataclass
class TrainState:
    model: Model
    optimiser: Optimizer
    scheduler: LRScheduler
    epoch: int = 0

    @property
    def _state_dict(self) -> dict[str, Any]:
        return {
            "model_state_dict": self.model.local_model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
        }

    @property
    def _model_state_dict(self) -> dict[str, Any]:
        return self.model.local_model.state_dict()

    def _load_state_dict(self, state_dict: dict[str, Any]) -> Self:
        self.model.local_model.load_state_dict(state_dict["model_state_dict"])
        self.optimiser.load_state_dict(state_dict["optimiser_state_dict"])
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        self.epoch = state_dict["epoch"]
        return self

    def save_state(self, path: Path) -> None:
        torch.save(self._state_dict, path)

    def load_state(self, path: Path) -> Self:
        state_dict = torch.load(path, weights_only=True)
        return self._load_state_dict(state_dict)

    def save_model(self, path: Path) -> None:
        torch.save(self._model_state_dict, path)
