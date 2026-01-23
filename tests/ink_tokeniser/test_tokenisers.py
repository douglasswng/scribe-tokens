"""Integration tests for tokenisers - verify tokenisation/detokenisation round-trips."""

from ink_tokeniser.discretes.abs import AbsTokeniser
from ink_tokeniser.discretes.rel import RelTokeniser
from ink_tokeniser.discretes.scribe import ScribeTokeniser
from ink_tokeniser.discretes.text import TextTokeniser
from schemas.parsed import Parsed


def test_abs_tokeniser():
    """Test absolute position tokeniser round-trip."""
    parsed = Parsed.load_random()
    digital_ink = parsed.ink

    tokeniser = AbsTokeniser()
    tokens = tokeniser.tokenise(digital_ink)
    detokenised = tokeniser.detokenise(tokens)

    # Verify we can visualize both
    digital_ink.visualise()
    detokenised.visualise()

    print(f"Abs tokeniser: {len(tokens)} tokens")


def test_rel_tokeniser():
    """Test relative position tokeniser round-trip."""
    parsed = Parsed.load_random()
    digital_ink = parsed.ink

    tokeniser = RelTokeniser()
    tokens = tokeniser.tokenise(digital_ink)
    detokenised = tokeniser.detokenise(tokens)

    digital_ink.visualise()
    detokenised.visualise()

    print(f"Rel tokeniser: {len(tokens)} tokens")


def test_scribe_tokeniser():
    """Test scribe tokeniser round-trip."""
    parsed = Parsed.load_random()
    digital_ink = parsed.ink

    tokeniser = ScribeTokeniser()
    tokens = tokeniser.tokenise(digital_ink)
    detokenised = tokeniser.detokenise(tokens)

    digital_ink.visualise()
    detokenised.visualise()

    print(f"Scribe tokeniser: {len(tokens)} tokens")


def test_text_tokeniser():
    """Test text tokeniser round-trip."""
    parsed = Parsed.load_random()
    digital_ink = parsed.ink

    tokeniser = TextTokeniser()
    tokens = tokeniser.tokenise(digital_ink)
    detokenised = tokeniser.detokenise(tokens)

    digital_ink.visualise()
    detokenised.visualise()

    print(f"Text tokeniser: {len(tokens)} tokens")


if __name__ == "__main__":
    test_abs_tokeniser()
    test_rel_tokeniser()
    test_scribe_tokeniser()
    test_text_tokeniser()
