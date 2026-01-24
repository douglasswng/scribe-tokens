import torch
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_

from ml_trainer.state import TrainState
from utils.distributed_context import distributed_context


class GradientHandler:
    """Handles gradient computation, reduction, and clipping."""

    def __init__(self, max_grad_norm: float):
        self._max_grad_norm = max_grad_norm

    def compute_gradients(self, model: Module, loss: torch.Tensor) -> torch.Tensor:
        """Compute gradients from loss and return max gradient norm for monitoring."""
        loss.backward()

        max_grad = torch.tensor(0.0, device=distributed_context.device)
        for param in model.parameters():
            if param.grad is not None:
                max_grad = torch.max(max_grad, param.grad.data.abs().max())

        return max_grad

    def clip_and_step(self, train_state: TrainState) -> None:
        """Clip gradients and apply optimizer/scheduler updates."""
        clip_grad_norm_(train_state.model.parameters(), max_norm=self._max_grad_norm)
        train_state.optimiser.step()
        train_state.scheduler.step()
