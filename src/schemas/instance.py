from dataclasses import dataclass

from torch import Tensor

from ink_repr.id import ReprId
from schemas.parsed import Parsed


@dataclass(frozen=True)
class Instance:
    parsed: Parsed
    repr_id: ReprId
    repr: Tensor
    char: Tensor

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
