from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerConfig:
    """Immutable configuration for training."""

    max_grad_norm: float = 1.0
    monitor_every_n_epochs: int = 1
    checkpoint_interval: int = 20
    max_checkpoints: int = 10
    patience: int = 10
