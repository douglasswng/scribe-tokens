from typing import Self
from dataclasses import dataclass, replace
from functools import cached_property
import random

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from core.utils.distributed_context import distributed_context
from core.data_schema.instance import Instance


@dataclass
class InstanceBatch:
    instances: list[Instance]

    @property
    def device(self) -> str:
        return distributed_context.device

    @property
    def repr(self) -> Tensor:
        return self._repr_and_pad[0].to(self.device)
    
    @property
    def repr_pad(self) -> Tensor:
        return self._repr_and_pad[1].to(self.device)
    
    @property
    def char(self) -> Tensor:
        return self._char_and_pad[0].to(self.device)
    
    @property
    def char_pad(self) -> Tensor:
        return self._char_and_pad[1].to(self.device)
    
    @property
    def writer(self) -> Tensor:
        return self._writer_and_pad[0].to(self.device)
    
    @property
    def repr_input(self) -> Tensor:
        return self.repr[:, :-1]
    
    @property
    def repr_target(self) -> Tensor:
        return self.repr[:, 1:]
    
    @property
    def repr_input_pad(self) -> Tensor:
        return self.repr_pad[:, :-1]
    
    @property
    def repr_target_pad(self) -> Tensor:
        return self.repr_pad[:, 1:]
    
    @cached_property
    def _repr_and_pad(self) -> tuple[Tensor, Tensor]:
        return self._stack_and_pad([instance.repr_tensor
                                    for instance in self.instances])

    @cached_property
    def _char_and_pad(self) -> tuple[Tensor, Tensor]:
        return self._stack_and_pad([instance.char_ids_tensor
                                    for instance in self.instances])
    
    @cached_property
    def _writer_and_pad(self) -> tuple[Tensor, Tensor]:
        return self._stack_and_pad([instance.writer_id_tensor
                                    for instance in self.instances])
    
    def _stack_and_pad(self, tensors: list[Tensor]) -> tuple[Tensor, Tensor]:
        stacked = pad_sequence(tensors, batch_first=True)
        
        pad = torch.zeros(stacked.shape[:2], dtype=torch.bool)
        for i, tensor in enumerate(tensors):
            pad[i, :tensor.shape[0]] = True
        
        return stacked, pad
    
    def get_sample(self, idx: int) -> Self:
        return replace(self, instances=[self.instances[idx]])
    
    def get_random_sample(self) -> Self:
        random_idx = random.randint(0, len(self.instances) - 1)
        return self.get_sample(random_idx)
    

@dataclass
class Batch:
    main_batch: InstanceBatch
    reference_batch: InstanceBatch | None

    def get_sample(self, idx: int) -> Self:
        if self.reference_batch is None:
            raise ValueError("Reference batch is required for batch.use_random_instance")
        
        assert len(self.main_batch.instances) == len(self.reference_batch.instances)

        return replace(self, main_batch=self.main_batch.get_sample(idx),
                       reference_batch=self.reference_batch.get_sample(idx))

    def get_random_sample(self) -> Self:
        random_idx = random.randint(0, len(self.main_batch.instances) - 1)
        return self.get_sample(random_idx)