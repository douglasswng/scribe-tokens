from typing import Self
from dataclasses import dataclass, replace
import random

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from core.data_schema.instance import Instance


@dataclass(frozen=True)
class InstancePair:
    main_instance: Instance
    ref_instance: Instance

    @property
    def context(self) -> Tensor:
        return torch.cat([self.ref_instance.repr, self.main_instance.char_input])
    
    @property
    def input(self) -> Tensor:
        return torch.cat([self.context, self.main_instance.repr_input])
    
    @property
    def target(self) -> Tensor:
        return torch.cat([self.context, self.main_instance.repr_target])
    
    @property
    def padding(self) -> Tensor:
        return torch.cat([torch.zeros_like(self.context), torch.ones_like(self.main_instance.repr_target)])
    
    @property
    def ref_repr_length(self) -> int:
        return self.ref_instance.repr.size(0)
    
    @property
    def main_char_input_length(self) -> int:
        return self.main_instance.char_input.size(0)


@dataclass(frozen=True)
class PairBatch:
    datapoints: list[InstancePair]
    
    @property
    def input(self) -> Tensor:
        inputs = [instance.input for instance in self.datapoints]
        return pad_sequence(inputs, batch_first=True, padding_value=0)
    
    @property
    def target(self) -> Tensor:
        targets = [instance.target for instance in self.datapoints]
        return pad_sequence(targets, batch_first=True, padding_value=0)
    
    @property
    def padding(self) -> Tensor:
        paddings = [instance.padding for instance in self.datapoints]
        return pad_sequence(paddings, batch_first=True, padding_value=0)
    
    @property
    def ref_repr_lengths(self) -> list[int]:
        return [instance.ref_repr_length for instance in self.datapoints]
    
    @property
    def main_char_input_lengths(self) -> list[int]:
        return [instance.main_char_input_length for instance in self.datapoints]

    def get_sample(self, idx: int) -> Self:
        return replace(self, datapoints=[self.datapoints[idx]])

    def get_random_sample(self) -> Self:
        random_idx = random.randint(0, len(self.datapoints) - 1)
        return self.get_sample(random_idx)
    

@dataclass(frozen=True)
class SingletonBatch:
    datapoints: list[Instance]


type Batch = PairBatch | SingletonBatch