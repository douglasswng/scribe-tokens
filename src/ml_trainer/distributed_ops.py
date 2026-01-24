import torch
import torch.distributed as dist

from utils.distributed_context import distributed_context


class DistributedOps:
    """Handles distributed tensor operations like reduction and synchronization."""

    @staticmethod
    def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Reduce tensor across all processes using average reduction."""
        if not distributed_context.is_distributed:
            return tensor

        if distributed_context.backend == "nccl":
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        elif distributed_context.backend == "gloo":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / distributed_context.world_size
        else:
            raise ValueError(f"Unsupported backend: {distributed_context.backend}")

        return tensor

    @staticmethod
    def reduce_max(tensor: torch.Tensor) -> torch.Tensor:
        """Reduce tensor across all processes using max reduction."""
        if not distributed_context.is_distributed:
            return tensor

        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor

    @staticmethod
    def reduce_losses(losses: dict[str, torch.Tensor]) -> None:
        """Reduce all loss tensors in-place."""
        if not distributed_context.is_distributed:
            return

        for loss_tensor in losses.values():
            DistributedOps.reduce_tensor(loss_tensor)
