from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerConfig:
    """Immutable configuration for training."""

    max_grad_norm: float = 1.0
    max_checkpoints: int = 3  # Number of most recent checkpoints to keep
    patience: int = 10
    grad_accum_steps: int = 1  # Number of gradient accumulation steps
