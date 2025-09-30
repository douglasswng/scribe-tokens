from typing import Self
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from core.data_schema.batch import Batch
from core.model.tracker import Tracker
from core.utils.distributed_context import distributed_context
from core.constants import HIDDEN_DIM, VOCAB_SIZE


class ModelMixin(ABC):
    @property
    @abstractmethod
    def local_model(self) -> nn.Module: ...

    @property
    @abstractmethod
    def num_params(self) -> int: ...

    @abstractmethod
    def losses(self, batch: Batch) -> dict[str, torch.Tensor]: ...
    
    @abstractmethod
    def monitor(self, batch: Batch, tracker: Tracker | None) -> None: ...

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        return self.losses(batch)


class LocalModel(ModelMixin, nn.Module):
    @property
    def local_model(self) -> Self:
        return self

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    @property
    def num_embedding_params(self) -> int:
        return VOCAB_SIZE * HIDDEN_DIM

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def init_weights(self):
        for module in self.modules():
            match module:
                case nn.Linear() | nn.Embedding():
                    torch.nn.init.normal_(module.weight, std=0.02)
                case nn.RMSNorm():
                    torch.nn.init.ones_(module.weight)
                case _:
                    pass


class DistributedModel(DDP):
    def __init__(self, local_model: LocalModel):
        if not distributed_context.is_distributed:
            raise ValueError("DistributedModel can only be used in distributed mode")
        
        super().__init__(local_model,
                         device_ids=distributed_context.device_ids,
                         find_unused_parameters=True)  # char embedder's unembed sometimes not used

    @property
    def local_model(self) -> LocalModel:
        return self.module

    @property
    def num_params(self) -> int:
        return self.local_model.num_params
    
    @property
    def num_embedding_params(self) -> int:
        return self.local_model.num_embedding_params
    
    def monitor(self, batch: Batch, tracker: Tracker | None) -> None:
        return self.local_model.monitor(batch, tracker)
    
    
type Model = LocalModel | DistributedModel