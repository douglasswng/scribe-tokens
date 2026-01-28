from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from constants import CHARS, NUM_CHARS
from ink_repr.factory import ReprFactory
from ink_repr.id import ReprId
from schemas.ink import DigitalInk
from schemas.parsed import Parsed


class IdMapper:
    _CHAR_ID_MAP = {char: id for id, char in enumerate(CHARS, 1)}
    _ID_CHAR_MAP = {v: k for k, v in _CHAR_ID_MAP.items()}
    _BOS_ID = NUM_CHARS + 1
    _EOS_ID = NUM_CHARS + 2

    @classmethod
    def chars_to_ids(cls, chars: list[str]) -> list[int]:
        return [cls._CHAR_ID_MAP[char] for char in chars]

    @classmethod
    def ids_to_chars(cls, ids: list[int]) -> list[str]:
        chars = []
        for id in ids:
            if id == cls._EOS_ID:
                break
            char = cls._ID_CHAR_MAP.get(id, "")
            chars.append(char)
        return chars

    @classmethod
    def str_to_ids(cls, s: str) -> list[int]:
        ids = cls.chars_to_ids(list(s))
        ids = [cls._BOS_ID] + ids + [cls._EOS_ID]
        return ids

    @classmethod
    def ids_to_str(cls, ids: list[int]) -> str:
        return "".join(cls.ids_to_chars(ids))


@dataclass(frozen=True)
class Instance:
    parsed: Parsed
    repr_id: ReprId
    repr: Tensor
    char: Tensor

    @classmethod
    def from_parsed(cls, parsed: Parsed, repr_id: ReprId) -> Self:
        repr = ReprFactory.from_ink(parsed.ink, repr_id=repr_id).to_tensor()
        char = torch.tensor(IdMapper.str_to_ids(parsed.text))
        return cls(parsed=parsed, repr_id=repr_id, repr=repr, char=char)

    @classmethod
    def from_ink(cls, ink: DigitalInk, repr_id: ReprId) -> Self:
        parsed = Parsed(ink=ink)
        return cls.from_parsed(parsed, repr_id)

    @classmethod
    def from_text(cls, text: str, repr_id: ReprId) -> Self:
        parsed = Parsed(text=text)
        return cls.from_parsed(parsed, repr_id)

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
