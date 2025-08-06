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
    _repr_tensor: Tensor

    @property
    def _device(self) -> str:
        return distributed_context.device

    @property
    def _repr(self) -> Tensor:
        return self._repr.to(self._device)
    
    @property
    def _char(self) -> Tensor:
        return torch.tensor(IdMapper.str_to_ids(self.parsed.text)).to(self._device)

    @property
    def _context(self) -> Tensor:
        match self.context_type:
            case 'repr':
                return self._repr
            case 'char':
                return self._char
            case None:
                return torch.tensor([])
    
    @property
    def _main(self) -> Tensor:
        match self.main_type:
            case 'repr':
                return self._repr
            case 'char':
                return self._char

    @property
    def _main_input(self) -> Tensor:
        return self._main[:-1]
    
    @property
    def _main_target(self) -> Tensor:
        return self._main[1:]

    @property
    def input(self) -> Tensor:
        return torch.cat([self._context, self._main_input], dim=0)

    @property
    def target(self) -> Tensor:
        return torch.cat([self._context, self._main_target], dim=0)

    @property
    def char_mask(self) -> Tensor:
        match self.context_type, self.main_type:
            case 'repr', 'char':
                zero_context = torch.zeros_like(self._context, dtype=torch.bool)
                one_main_input = torch.ones_like(self._main_input, dtype=torch.bool)
                return torch.cat([zero_context, one_main_input], dim=0)
            case 'char', 'repr':
                return torch.ones_like(self._context, dtype=torch.bool)
            case _:
                raise ValueError(f'Invalid context and main types: {self.context_type}, {self.main_type}')

    @property
    def target_mask(self) -> Tensor:
        zero_context = torch.zeros_like(self._context, dtype=torch.bool)
        one_main_target = torch.ones_like(self._main_target, dtype=torch.bool)
        return torch.cat([zero_context, one_main_target], dim=0)