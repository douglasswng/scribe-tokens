from torch import Tensor

from core.data_schema import DigitalInk
from core.repr import Repr, ReprFactory, ReprId, TokenReprId, VectorReprId, VectorReprType
from repr.reprs.point3 import Point3Repr
from repr.reprs.point5 import Point5Repr
from repr.reprs.tokens import TokenRepr


class DefaultReprFactory(ReprFactory):
    @classmethod
    def _get_vector_repr(cls, id: VectorReprId) -> type[Repr]:
        match id.type:
            case VectorReprType.POINT3:
                return Point3Repr
            case VectorReprType.POINT5:
                return Point5Repr
            case _:
                raise ValueError(f"Invalid vector repr type: {id.type}")

    @classmethod
    def ink_to_repr(cls, id: ReprId, ink: DigitalInk) -> Repr:
        match id:
            case VectorReprId():
                return cls._get_vector_repr(id).from_ink(id, ink)
            case TokenReprId():
                return TokenRepr.from_ink(id, ink)
            case _:
                raise ValueError(f"Invalid repr id: {id}")

    @classmethod
    def tensor_to_repr(cls, id: ReprId, tensor: Tensor) -> Repr:
        match id:
            case VectorReprId():
                return cls._get_vector_repr(id).from_tensor(id, tensor)
            case TokenReprId():
                return TokenRepr.from_tensor(id, tensor)
            case _:
                raise ValueError(f"Invalid repr id: {id}")


if __name__ == "__main__":
    import torch

    from core.data_schema import Parsed

    parsed = Parsed.load_random()
    ink = parsed.ink
    ink.visualise()

    for repr_id in TokenReprId.create_defaults() + VectorReprId.create_defaults():
        print(f"Repr ID: {repr_id}")
        repr = DefaultReprFactory.ink_to_repr(repr_id, ink)

        print(str(repr)[:1000])
        print()

        tensor1 = DefaultReprFactory.repr_to_tensor(repr_id, repr)
        tensor2 = DefaultReprFactory.ink_to_tensor(repr_id, ink)
        assert torch.equal(tensor1, tensor2)

        ink1 = DefaultReprFactory.repr_to_ink(repr_id, repr)
        ink2 = DefaultReprFactory.tensor_to_ink(repr_id, tensor1)

        print(ink1)
        ink1.visualise()

        raise
