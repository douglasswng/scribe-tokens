from typing import Self
from dataclasses import dataclass, replace
import random
from functools import cached_property

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from core.data_schema.instance import Instance


@dataclass(frozen=True)
class InstancePair:
    main_instance: Instance
    ref_instance: Instance

    @property
    def char_mask(self) -> Tensor:
        char_mask = torch.zeros_like(self.context, dtype=torch.bool)
        repr_len = len(self.ref_instance.repr)
        char_mask[repr_len:] = True
        return char_mask

    @cached_property
    def context(self) -> Tensor:
        return torch.cat([self.ref_instance.repr, self.main_instance.char_input])

    @property
    def input(self) -> Tensor:
        return torch.cat([self.context, self.main_instance.repr_input])

    @cached_property
    def _target(self) -> Tensor:
        return torch.cat([self.context, self.main_instance.repr_target])

    @property
    def _target_mask(self) -> Tensor:
        target_mask = torch.ones_like(self._target, dtype=torch.bool)
        context_len = len(self.context)
        target_mask[context_len:] = False
        return target_mask


@dataclass(frozen=True)
class PairBatch:
    datapoints: list[InstancePair]
    
    @property
    def input(self) -> Tensor:
        inputs = [instance.input for instance in self.datapoints]
        return self._pad_sequence(inputs)

    @property
    def char_mask(self) -> Tensor:
        char_masks = [instance.char_mask for instance in self.datapoints]
        return self._pad_sequence(char_masks)

    @property
    def target(self) -> Tensor:
        targets = [instance._target for instance in self.datapoints]
        return self._pad_sequence(targets)

    @property
    def target_mask(self) -> Tensor:
        target_masks = [instance._target_mask for instance in self.datapoints]
        return self._pad_sequence(target_masks)

    def get_sample(self, idx: int) -> Self:
        return replace(self, datapoints=[self.datapoints[idx]])

    def get_random_sample(self) -> Self:
        random_idx = random.randint(0, len(self.datapoints) - 1)
        return self.get_sample(random_idx)

    def _pad_sequence(self, tensors: list[Tensor]) -> Tensor:
        return pad_sequence(tensors, batch_first=True, padding_value=0)
    

@dataclass(frozen=True)
class SingletonBatch:
    datapoints: list[Instance]


type Batch = PairBatch | SingletonBatch