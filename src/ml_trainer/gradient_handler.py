import torch
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_

from ml_trainer.state import TrainState
from utils.distributed_context import distributed_context


class GradientHandler:
    """Handles gradient monitoring and clipping."""

    def __init__(self, max_grad_norm: float):
        self._max_grad_norm = max_grad_norm

    def get_max_grad(self, model: Module) -> torch.Tensor:
        """Get maximum gradient norm across all parameters."""
        max_grad = torch.tensor(0.0, device=distributed_context.device)
        for param in model.parameters():
            if param.grad is not None:
                max_grad = torch.max(max_grad, param.grad.data.abs().max())
        return max_grad

    def clip(self, train_state: TrainState) -> None:
        """Clip gradients to prevent exploding gradients."""
        clip_grad_norm_(train_state.model.parameters(), max_norm=self._max_grad_norm)
