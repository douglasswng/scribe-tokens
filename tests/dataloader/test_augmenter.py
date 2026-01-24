"""Test data augmentation with visualization."""

from dataloader.augmenter import Augmenter
from ink_repr.factory import ReprFactory
from ink_repr.id import TokeniserId
from schemas.parsed import Parsed


def test_augmentation():
    """Test that augmentation produces valid transformed data."""
    parsed = Parsed.load_random()
    parsed.visualise()

    augmented_parsed = Augmenter.augment(parsed)
    augmented_parsed.visualise()

    # Test with tokeniser
    repr_id = TokeniserId.create_defaults()[0]

    original_tensor = ReprFactory.from_ink(parsed.ink, repr_id).to_tensor()
    augmented_tensor = ReprFactory.from_ink(augmented_parsed.ink, repr_id).to_tensor()

    # Convert back to ink and visualize
    original_ink = ReprFactory.from_tensor(original_tensor, repr_id).to_ink()
    augmented_ink = ReprFactory.from_tensor(augmented_tensor, repr_id).to_ink()
    original_ink.visualise()
    augmented_ink.visualise()

    print("Ink length:", len(parsed.ink))
    print("Original token count:", original_tensor.size(0))
    print("Augmented token count:", augmented_tensor.size(0))


if __name__ == "__main__":
    test_augmentation()
