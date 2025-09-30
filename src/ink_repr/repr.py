from typing import Protocol, Self

from torch import Tensor

from schemas.ink import DigitalInk


class InkRepr(Protocol):
    def __str__(self) -> str: ...

    @classmethod
    def from_ink(cls, ink: DigitalInk) -> Self: ...

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> Self: ...

    def to_ink(self) -> DigitalInk: ...
    def to_tensor(self) -> Tensor: ...
