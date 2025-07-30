from typing import Callable
from dataclasses import dataclass
from functools import cached_property

import torch
from torch import Tensor

from core.data_schema.ink import DigitalInk
from core.data_schema.parsed import Parsed
from core.data_schema.utils import IdMapper
from core.utils.distributed_context import distributed_context


type ReprCallable = Callable[[DigitalInk], Tensor]


@dataclass(frozen=True)
class Instance:
    parsed: Parsed
    repr_callable: ReprCallable

    @property
    def device(self) -> str:
        return distributed_context.device

    @cached_property
    def repr(self) -> Tensor:
        return self.repr_callable(self.parsed.ink).to(self.device)
    
    @cached_property
    def writer(self) -> Tensor:
        return torch.tensor([IdMapper.writer_to_id(self.parsed.writer)]).to(self.device)
    
    @cached_property
    def char(self) -> Tensor:
        return torch.tensor(IdMapper.str_to_ids(self.parsed.text)).to(self.device)
    
    @property
    def repr_input(self) -> Tensor:
        return self.repr[:-1]
    
    @property
    def repr_target(self) -> Tensor:
        return self.repr[1:]
    
    @property
    def char_input(self) -> Tensor:
        return self.char
    
    @property
    def char_target(self) -> Tensor:
        return self.char[1:-1]  # drop bos and eos