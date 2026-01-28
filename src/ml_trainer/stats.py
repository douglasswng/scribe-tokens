import time
from typing import Literal

from pydantic import BaseModel, Field


def add_prefix(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in stats.items()}


def drop_inf(stats: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in stats.items() if v != float("inf")}


class BatchStats(BaseModel):
    losses: dict[str, float]
    max_grad: float = float("inf")
    lr: float = float("inf")

    @property
    def loss(self) -> float:
        return sum(self.losses.values())

    @property
    def summary_dict(self) -> dict[str, str]:
        return {
            "loss": f"{self.loss:.4f}",
            "max_grad": f"{self.max_grad:.4f}",
            "lr": f"{self.lr:.4f}",
        }

    @property
    def full_dict(self) -> dict[str, float]:
        dict = {**self.losses, "max_grad": self.max_grad, "lr": self.lr}
        if len(self.losses) > 1:  # Only add loss if there are multiple losses
            dict["loss"] = self.loss
        return add_prefix("batch", drop_inf(dict))


class EpochStats(BaseModel):
    type: Literal["train", "val"]
    epoch: int
    batch_stats: list[BatchStats] = Field(default_factory=list)

    @property
    def curr_batch_stats(self) -> BatchStats:
        return self.batch_stats[-1]

    @property
    def loss(self) -> float:
        return sum(batch_stats.loss for batch_stats in self.batch_stats) / len(self.batch_stats)

    @property
    def losses(self) -> dict[str, float]:
        num_batches = len(self.batch_stats)
        return {
            k: sum(batch_stats.losses[k] for batch_stats in self.batch_stats) / num_batches
            for k in self.batch_stats[0].losses
        }

    @property
    def max_grad(self) -> float:
        return max(batch_stats.max_grad for batch_stats in self.batch_stats)

    @property
    def full_dict(self) -> dict[str, float]:
        dict = {**self.losses, "max_grad": self.max_grad}
        if len(self.losses) > 1:  # Only add loss if there are multiple losses
            dict["loss"] = self.loss
        return add_prefix(f"{self.type}_epoch", drop_inf(dict))


class TrainStats(BaseModel):
    train_epoch_stats: list[EpochStats] = Field(default_factory=list)
    val_epoch_stats: list[EpochStats] = Field(default_factory=list)
    start_time: float = Field(default_factory=time.time)

    @property
    def time_ellapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def time_ellapsed_formatted(self) -> str:
        total_seconds = int(self.time_ellapsed)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @property
    def curr_epoch_stats_formatted(self) -> str:
        return (
            f"Time Elapsed: {self.time_ellapsed_formatted}\n"
            f"Train Loss: {self.curr_train_epoch_stats.loss:.4f}\n"
            f"Val Loss: {self.curr_val_epoch_stats.loss:.4f}\n"
            f"Max Grad: {self.curr_train_epoch_stats.max_grad:.4f}\n"
        )

    @property
    def curr_train_epoch_stats(self) -> EpochStats:
        return self.train_epoch_stats[-1]

    @property
    def curr_val_epoch_stats(self) -> EpochStats:
        return self.val_epoch_stats[-1]

    @property
    def curr_train_batch_stats(self) -> BatchStats:
        return self.curr_train_epoch_stats.curr_batch_stats

    @property
    def curr_val_batch_stats(self) -> BatchStats:
        return self.curr_val_epoch_stats.curr_batch_stats

    def new_epoch(self, epoch: int) -> None:
        self.train_epoch_stats.append(EpochStats(type="train", epoch=epoch))
        self.val_epoch_stats.append(EpochStats(type="val", epoch=epoch))

    def add_train_batch_stats(self, batch_stats: BatchStats) -> None:
        self.curr_train_epoch_stats.batch_stats.append(batch_stats)

    def add_val_batch_stats(self, batch_stats: BatchStats) -> None:
        self.curr_val_epoch_stats.batch_stats.append(batch_stats)
