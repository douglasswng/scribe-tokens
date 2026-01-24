"""Integration test for tokeniser factory."""

from ink_tokeniser.factory import TokeniserFactory
from ink_tokeniser.id import TokeniserId
from schemas.parsed import Parsed


def test_tokeniser_factory():
    """Test tokeniser factory with all default tokenisers."""
    ink = Parsed.load_random().ink
    ink.visualise()
    print(f"DigitalInk length: {len(ink)}")

    for id in TokeniserId.create_defaults():
        tokeniser = TokeniserFactory.create(id)
        tokens = tokeniser.tokenise(ink)
        print(f"{id} length: {len(tokens)}")

        detokenised_ink = tokeniser.detokenise(tokens)
        detokenised_ink.visualise()


if __name__ == "__main__":
    test_tokeniser_factory()
