from typing import Protocol, Self

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from schemas.batch import Batch
from utils.distributed_context import distributed_context


class TrainableModel(Protocol):  # methods needed by the trainer
    @property
    def local_model(self) -> nn.Module: ...  # for saving since dont want to save DDP model

    def losses(self, batch: Batch) -> dict[str, Tensor]: ...

    def monitor(self, batch: Batch) -> None: ...

    @property
    def num_params(self) -> int: ...


class LocalModel(TrainableModel, nn.Module):
    @property
    def local_model(self) -> Self:
        return self

    def init_weights(self):
        for module in self.modules():
            match module:
                case nn.Linear() | nn.Embedding():
                    torch.nn.init.normal_(module.weight, std=0.02)
                case nn.RMSNorm():
                    torch.nn.init.ones_(module.weight)
                case _:
                    pass

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DDPModel(TrainableModel, DDP):
    def __init__(self, local_model: LocalModel):
        if not distributed_context.is_distributed:
            raise ValueError("DistributedModel can only be used in distributed mode")

        super().__init__(local_model, device_ids=distributed_context.device_ids)

    @property
    def local_model(self) -> LocalModel:
        return self.module

    def losses(self, batch: Batch) -> dict[str, Tensor]:
        return self.local_model.losses(batch)

    def monitor(self, batch: Batch) -> None:
        return self.local_model.monitor(batch)

    @property
    def num_params(self) -> int:
        return self.local_model.num_params


type Model = LocalModel | DDPModel
