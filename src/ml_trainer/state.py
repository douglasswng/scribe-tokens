from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ml_model.model import Model


@dataclass
class TrainState:
    """Pure data container for training state."""

    model: Model
    optimiser: Optimizer
    scheduler: LRScheduler
    epoch: int = 0
