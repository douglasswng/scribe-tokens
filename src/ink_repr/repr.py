from typing import Protocol

from torch import Tensor

from schemas.ink import DigitalInk


class InkRepr(Protocol):
    def __str__(self) -> str: ...

    def to_ink(self) -> DigitalInk: ...
    def to_tensor(self) -> Tensor: ...
