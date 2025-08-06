from typing import Self
from dataclasses import dataclass, replace
import random
from functools import cached_property

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from core.data_schema.instance import Instance


@dataclass(frozen=True)
class Batch:
    instances: list[Instance]

    @property
    def input(self) -> Tensor:
        inputs = [instance.input for instance in self.instances]
        return self._pad_sequence(inputs)

    @property
    def target(self) -> Tensor:
        targets = [instance.target for instance in self.instances]
        return self._pad_sequence(targets)

    @property
    def char_mask(self) -> Tensor:
        char_masks = [instance.char_mask for instance in self.instances]
        return self._pad_sequence(char_masks)

    @property
    def target_mask(self) -> Tensor:
        target_masks = [instance.target_mask for instance in self.instances]
        return self._pad_sequence(target_masks)

    def get_sample(self, idx: int) -> Self:
        return replace(self, instances=[self.instances[idx]])

    def get_random_sample(self) -> Self:
        random_idx = random.randint(0, len(self.instances) - 1)
        return self.get_sample(random_idx)

    def _pad_sequence(self, tensors: list[Tensor]) -> Tensor:
        return pad_sequence(tensors, batch_first=True, padding_value=0)