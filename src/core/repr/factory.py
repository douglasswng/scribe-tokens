from typing import Protocol

from torch import Tensor

from core.data_schema import DigitalInk
from core.repr.id import ReprId
from core.repr.repr import Repr


class ReprFactory(Protocol):
    @classmethod
    def ink_to_repr(cls, id: ReprId, ink: DigitalInk) -> Repr: ...

    @classmethod
    def tensor_to_repr(cls, id: ReprId, tensor: Tensor) -> Repr: ...

    @classmethod
    def repr_to_ink(cls, id: ReprId, repr: Repr) -> DigitalInk:
        return repr.to_ink(id)

    @classmethod
    def repr_to_tensor(cls, id: ReprId, repr: Repr) -> Tensor:
        return repr.to_tensor(id)

    @classmethod
    def ink_to_tensor(cls, id: ReprId, ink: DigitalInk) -> Tensor:
        repr = cls.ink_to_repr(id, ink)
        return cls.repr_to_tensor(id, repr)

    @classmethod
    def tensor_to_ink(cls, id: ReprId, tensor: Tensor) -> DigitalInk:
        repr = cls.tensor_to_repr(id, tensor)
        return cls.repr_to_ink(id, repr)
