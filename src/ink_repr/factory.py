from torch import Tensor

from ink_repr.id import ReprId, TokeniserId, VectorReprId, VectorType
from ink_repr.repr import InkRepr
from ink_repr.reprs.point3 import Point3Repr
from ink_repr.reprs.point5 import Point5Repr
from ink_repr.reprs.token import TokenRepr
from ink_tokeniser.factory import TokeniserFactory
from schemas.ink import DigitalInk


class ReprFactory:
    @classmethod
    def from_ink(cls, ink: DigitalInk, repr_id: ReprId) -> InkRepr:
        """Create a representation from digital ink based on ID."""
        match repr_id:
            case VectorReprId(type=VectorType.POINT3):
                return Point3Repr.from_ink(ink)
            case VectorReprId(type=VectorType.POINT5):
                return Point5Repr.from_ink(ink)
            case TokeniserId():
                tokeniser = TokeniserFactory.create(repr_id)
                return TokenRepr.from_ink(ink, tokeniser)
            case _:
                raise ValueError(f"Unknown representation ID: {repr_id}")

    @classmethod
    def from_tensor(cls, tensor: Tensor, repr_id: ReprId) -> InkRepr:
        """Create a representation from tensor based on ID."""
        match repr_id:
            case VectorReprId(type=VectorType.POINT3):
                return Point3Repr.from_tensor(tensor)
            case VectorReprId(type=VectorType.POINT5):
                return Point5Repr.from_tensor(tensor)
            case TokeniserId():
                tokeniser = TokeniserFactory.create(repr_id)
                return TokenRepr.from_tensor(tensor, tokeniser)
            case _:
                raise ValueError(f"Unknown representation ID: {repr_id}")


if __name__ == "__main__":
    import torch

    from schemas.parsed import Parsed

    parsed = Parsed.load_random()
    ink = parsed.ink
    ink.visualise()

    for repr_id in TokeniserId.create_defaults() + VectorReprId.create_defaults():
        print(f"Repr ID: {repr_id}")
        repr = ReprFactory.from_ink(ink, repr_id)

        print(str(repr)[:1000])
        print()

        tensor1 = repr.to_tensor()
        tensor2 = ReprFactory.from_ink(ink, repr_id).to_tensor()
        assert torch.equal(tensor1, tensor2)

        ink1 = repr.to_ink()
        ink2 = ReprFactory.from_tensor(tensor1, repr_id).to_ink()

        print(ink1)
        ink1.visualise()

        raise
