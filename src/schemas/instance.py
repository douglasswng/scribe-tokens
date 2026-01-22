from dataclasses import dataclass
from functools import cached_property

import torch
from torch import Tensor

from constants import CHARS, NUM_CHARS
from schemas.parsed import Parsed
from utils.distributed_context import distributed_context


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


class IdMapper:
    _CHAR_ID_MAP = {char: id for id, char in enumerate(CHARS, 1)}
    _ID_CHAR_MAP = {v: k for k, v in _CHAR_ID_MAP.items()}

    @classmethod
    def chars_to_ids(cls, chars: list[str]) -> list[int]:
        return [cls._CHAR_ID_MAP[char] for char in chars]

    @classmethod
    def ids_to_chars(cls, ids: list[int]) -> list[str]:
        return [cls._ID_CHAR_MAP.get(id, "") for id in ids]

    @classmethod
    def str_to_ids(cls, s: str) -> list[int]:
        bos_id = NUM_CHARS + 1
        eos_id = NUM_CHARS + 2

        ids = cls.chars_to_ids(list(s))
        ids = [bos_id] + ids + [eos_id]
        return ids

    @classmethod
    def ids_to_str(cls, ids: list[int]) -> str:
        return "".join(cls.ids_to_chars(ids))
