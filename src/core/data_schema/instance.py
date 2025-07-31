from typing import Literal
from dataclasses import dataclass
from functools import cached_property

import torch
from torch import Tensor

from core.data_schema.parsed import Parsed
from core.data_schema.utils import IdMapper
from core.utils.distributed_context import distributed_context


@dataclass(frozen=True)
class Instance:
    parsed: Parsed
    context_type: Literal['repr', 'char', None]  # for HTR, context is repr, for HTG, context is char, for pretraining (NTP), context is None
    main_type: Literal['repr', 'char']  # for HTR, main is char, for HTG, main is repr, for pretraining (NTP), main is repr
    _repr: Tensor

    @property
    def device(self) -> str:
        return distributed_context.device

    @property
    def repr(self) -> Tensor:
        return self._repr.to(self.device)
    
    @property
    def char(self) -> Tensor:
        return torch.tensor(IdMapper.str_to_ids(self.parsed.text)).to(self.device)

    @property
    def context(self) -> Tensor:
        match self.context_type:
            case 'repr':
                return self.repr
            case 'char':
                return self.char
            case None:
                return torch.tensor([])
    
    @property
    def main(self) -> Tensor:
        match self.main_type:
            case 'repr':
                return self.repr
            case 'char':
                return self.char

    @property
    def main_input(self) -> Tensor:
        return self.main[:-1]
    
    @property
    def main_target(self) -> Tensor:
        return self.main[1:]

    @property
    def input(self) -> Tensor:
        return torch.cat([self.context, self.main_input], dim=0)

    @property
    def target(self) -> Tensor:
        return torch.cat([self.context, self.main_target], dim=0)

    @property
    def char_mask(self) -> Tensor:
        match self.context_type, self.main_type:
            case 'repr', 'char':
                zero_context = torch.zeros_like(self.context, dtype=torch.bool)
                one_main_input = torch.ones_like(self.main_input, dtype=torch.bool)
                return torch.cat([zero_context, one_main_input], dim=0)
            case 'char', 'repr':
                return torch.ones_like(self.context, dtype=torch.bool)
            case _:
                raise ValueError(f'Invalid context and main types: {self.context_type}, {self.main_type}')

    @property
    def target_mask(self) -> Tensor:
        zero_context = torch.zeros_like(self.context, dtype=torch.bool)
        one_main_target = torch.ones_like(self.main_target, dtype=torch.bool)
        return torch.cat([zero_context, one_main_target], dim=0)