from typing import Protocol, Self

from torch import Tensor

from core.data_schema import DigitalInk
from core.repr.id import ReprId


class Repr(Protocol):
    def __str__(self) -> str: ...

    @classmethod
    def from_ink(cls, id: ReprId, ink: DigitalInk) -> Self: ...

    @classmethod
    def from_tensor(cls, id: ReprId, tensor: Tensor) -> Self: ...

    def to_ink(self, id: ReprId) -> DigitalInk: ...
    def to_tensor(self, id: ReprId) -> Tensor: ...
