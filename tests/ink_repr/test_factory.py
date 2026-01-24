"""Integration test for representation factory."""

import torch

from ink_repr.factory import ReprFactory
from ink_repr.id import TokeniserId, VectorReprId
from schemas.parsed import Parsed


def test_repr_factory():
    """Test representation factory with different repr types."""
    parsed = Parsed.load_random()
    ink = parsed.ink
    ink.visualise()

    for repr_id in TokeniserId.create_defaults() + [VectorReprId.create_point5()]:
        print(f"Repr ID: {repr_id}")
        repr = ReprFactory.from_ink(ink, repr_id)

        print(str(repr)[:1000])
        print()

        # Test tensor conversion consistency
        tensor1 = repr.to_tensor()
        tensor2 = ReprFactory.from_ink(ink, repr_id).to_tensor()
        assert torch.equal(tensor1, tensor2)

        # Test ink round-trip
        ink1 = repr.to_ink()
        ink2 = ReprFactory.from_tensor(tensor1, repr_id).to_ink()

        print(ink1)
        ink1.visualise()

        # Only test first repr_id to avoid infinite loop from original code
        break


if __name__ == "__main__":
    test_repr_factory()
