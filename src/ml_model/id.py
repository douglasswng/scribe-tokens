from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

from constants import (
    CHECKPOINTS_DIR,
    GRPO_MONITOR_EVERY,
    GRPO_STEPS,
    MODELS_DIR,
    NUM_EPOCHS,
    PATIENCE_FACTOR,
)
from ink_repr.id import ReprId, TokeniserId, VectorReprId


class Task(StrEnum):
    HTR = "HTR"
    HTG = "HTG"
    NTP = "NTP"
    HTR_SFT = "HTR_SFT"
    HTG_SFT = "HTG_SFT"
    HTG_GRPO = "HTG_GRPO"

    @property
    def need_init_weights(self) -> bool:
        return self in {
            Task.HTG,
            Task.HTR,
            Task.NTP,
        }  # other tasks load from pretrained

    @property
    def num_epochs(self) -> int:
        match self:
            case Task.HTG_GRPO:
                return GRPO_STEPS // GRPO_MONITOR_EVERY
            case _:
                return NUM_EPOCHS

    @property
    def patience(self) -> int:
        return int(self.num_epochs * PATIENCE_FACTOR)


@dataclass(frozen=True)
class ModelId:
    task: Task
    repr_id: ReprId

    def __str__(self) -> str:
        return f"Task: {self.task}, Repr: {self.repr_id}"

    @classmethod
    def _get_repr_ids(cls) -> list[ReprId]:
        return [
            # skip point-3 cannot be used for generation
            # skip absolute tokeniser since vocab size is too large
            TokeniserId.create_scribe(),
            TokeniserId.create_rel(),
            TokeniserId.create_text(),
            VectorReprId.create_point5(),
        ]

    @classmethod
    def create_task_model_ids(cls, task: Task) -> list[Self]:
        return [cls(task=task, repr_id=repr_id) for repr_id in cls._get_repr_ids()]

    @classmethod
    def create_defaults(cls) -> list[Self]:
        model_ids = []
        for repr_id in cls._get_repr_ids():
            for task in Task:
                model_ids.append(cls(task=task, repr_id=repr_id))
        return model_ids

    @property
    def _base_dir(self) -> Path:
        return Path(self.task) / self.repr_id.type

    @property
    def checkpoint_dir(self) -> Path:
        return CHECKPOINTS_DIR / self._base_dir

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / self._base_dir / "best.pt"

    def list_checkpoint_paths(self) -> list[Path]:
        return list(self.checkpoint_dir.glob("checkpoint_*.pt"))

    def get_latest_checkpoint_path(self) -> Path | None:
        checkpoint_paths = self.list_checkpoint_paths()
        if not checkpoint_paths:
            return None

        latest_path = max(checkpoint_paths, key=lambda p: int(p.stem.split("_")[1]))
        return latest_path

    def get_checkpoint_path(self, epoch: int) -> Path:
        return self.checkpoint_dir / f"checkpoint_{epoch}.pt"
