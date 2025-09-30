from dataclasses import dataclass
from functools import cached_property

import torch
from core.data_schema.parsed import Parsed
from core.utils.distributed_context import distributed_context
from torch import Tensor

from core.data_schema.utils import IdMapper


@dataclass(frozen=True)
class Instance:
    parsed: Parsed
    _repr_tensor: Tensor

    @cached_property
    def _device(self) -> str:
        return distributed_context.device

    @cached_property
    def repr(self) -> Tensor:
        return self._repr_tensor.to(self._device)

    @cached_property
    def char(self) -> Tensor:
        return torch.tensor(IdMapper.str_to_ids(self.parsed.text)).to(self._device)

    @property
    def repr_input(self) -> Tensor:
        return self.repr[:-1]

    @property
    def repr_target(self) -> Tensor:
        return self.repr[1:]

    @property
    def char_input(self) -> Tensor:
        return self.char[:-1]

    @property
    def char_target(self) -> Tensor:
        return self.char[1:]

    @property
    def char_bos(self) -> Tensor:
        return self.char[0]

    @property
    def char_eos(self) -> Tensor:
        return self.char[-1]

    @property
    def repr_bos(self) -> Tensor:
        return self.repr[0]

    @property
    def repr_eos(self) -> Tensor:
        return self.repr[-1]
