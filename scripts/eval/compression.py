"""Evaluation script for tokeniser compression rate."""

import csv
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from constants import DATASET, RESULTS_DIR
from dataloader.split import create_datasplit
from ink_tokeniser.factory import TokeniserFactory
from ink_tokeniser.id import TokeniserId, TokenType
from ink_tokeniser.tokeniser import Tokeniser
from schemas.ink import DigitalInk
from schemas.parsed import Parsed

assert DATASET == "iam", "Use IAM for compression rate"

RESULTS_PATH = RESULTS_DIR / "compression.csv"
TOKEN_TYPES = [TokenType.SCRIBE, TokenType.REL, TokenType.ABS, TokenType.TEXT]
DELTAS = [1, 2, 4, 8, 16, 32]
VOCAB_SIZES = [
    5000,
    10000,
    15000,
    20000,
    25000,
    30000,
    35000,
    40000,
    45000,
    50000,
    55000,
    60000,
    65000,
    70000,
    75000,
    80000,
    85000,
    90000,
    95000,
    100000,
]


def create_tokeniser_ids() -> Iterator[TokeniserId]:
    """Iterate over all tokeniser IDs used for compression evaluation."""
    for token_type in TOKEN_TYPES:
        for delta in DELTAS:
            for vocab_size in VOCAB_SIZES:
                yield TokeniserId(type=token_type, delta=delta, vocab_size=vocab_size)


@lru_cache(maxsize=1)
def get_test_paths() -> tuple[Path, ...]:
    """Get parsed test paths from the dataset split."""
    data_split = create_datasplit()
    return tuple(data_split.test_paths)


def get_test_ink_iterator() -> Iterator[DigitalInk]:
    """Iterate over test inks in origin-normalized coordinates."""
    for path in get_test_paths():
        yield Parsed.from_path(path).ink.to_origin()


def get_tokeniser_key(tokeniser_id: TokeniserId) -> tuple[str, str, str]:
    """Create a stable CSV key for a tokeniser ID."""
    return (tokeniser_id.type.value, str(tokeniser_id.delta), str(tokeniser_id.vocab_size))


def load_existing_keys(results_path: Path) -> set[tuple[str, str, str]]:
    """Load already evaluated tokeniser keys from the results CSV."""
    existing_keys: set[tuple[str, str, str]] = set()
    if not results_path.exists():
        return existing_keys

    with open(results_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokeniser_type = row.get("tokeniser_type")
            delta = row.get("delta")
            vocab_size = row.get("vocab_size")
            if tokeniser_type and delta and vocab_size:
                existing_keys.add((tokeniser_type, delta, vocab_size))
    return existing_keys


def should_skip_evaluation(
    tokeniser_id: TokeniserId, existing_keys: set[tuple[str, str, str]]
) -> bool:
    """Check if a tokeniser should be skipped."""
    if not tokeniser_id.tokeniser_path.exists():
        print(f"Skipping {tokeniser_id}: tokeniser path not found")
        return True

    if not tokeniser_id.vocab_path.exists() or not tokeniser_id.merges_path.exists():
        print(f"Skipping {tokeniser_id}: missing vocab or merges file")
        return True

    if get_tokeniser_key(tokeniser_id) in existing_keys:
        print(f"Skipping {tokeniser_id}: already evaluated")
        return True

    return False


def add_metric(results_path: Path, tokeniser_id: TokeniserId, compression_rate: float) -> None:
    """Append one compression result to the CSV."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["tokeniser_type", "delta", "vocab_size", "compression_rate"]
    row = [tokeniser_id.type.value, tokeniser_id.delta, tokeniser_id.vocab_size, compression_rate]

    if results_path.exists():
        with open(results_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    else:
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(row)

    print(f"Added {tokeniser_id}: compression_rate={compression_rate}")


def calculate_compression_rate(tokeniser: Tokeniser) -> float:
    """Calculate average compression rate over the test split."""
    compression_rates: list[float] = []

    for ink in get_test_ink_iterator():
        tokens = tokeniser.tokenise(ink)
        if len(tokens) == 0:
            continue
        compression_rates.append(len(ink) / len(tokens))

    return sum(compression_rates) / len(compression_rates) if compression_rates else 0.0


def evaluate_tokeniser(tokeniser_id: TokeniserId, existing_keys: set[tuple[str, str, str]]) -> None:
    """Evaluate one tokeniser and write its compression rate."""
    if should_skip_evaluation(tokeniser_id, existing_keys):
        return

    try:
        tokeniser = TokeniserFactory.create(tokeniser_id)
        compression_rate = calculate_compression_rate(tokeniser)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Skipping {tokeniser_id}: {exc}")
        return

    add_metric(RESULTS_PATH, tokeniser_id, compression_rate)
    existing_keys.add(get_tokeniser_key(tokeniser_id))


def main() -> None:
    """Evaluate all tokenisers for compression rate."""
    if len(get_test_paths()) == 0:
        print("No test samples found. Skipping compression evaluation.")
        return

    existing_keys = load_existing_keys(RESULTS_PATH)
    for tokeniser_id in create_tokeniser_ids():
        evaluate_tokeniser(tokeniser_id, existing_keys)


if __name__ == "__main__":
    main()
