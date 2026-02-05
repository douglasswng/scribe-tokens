from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from constants import PARSED_DIR, TEST_SPLIT_PATH, TRAIN_SPLIT_PATH, VAL_SPLIT_PATH


def load_split_filenames(split_file: Path) -> set[str]:
    """Load the exact filenames from a split file."""
    if not split_file.exists():
        return set()
    with open(split_file, "r") as f:
        # Read and strip whitespace, filter empty lines
        return {line.strip() for line in f if line.strip()}


def load_split(split_file: Path) -> list[Path]:
    stems = load_split_filenames(split_file)
    parsed_paths = list(PARSED_DIR.glob("*.json"))
    return [p for p in parsed_paths if p.stem in stems]


@lru_cache(maxsize=1)
def split_parsed_paths() -> tuple[list[Path], list[Path], list[Path]]:
    """Split parsed paths based on predefined split files with exact filename matching."""
    train_paths = load_split(TRAIN_SPLIT_PATH)
    val_paths = load_split(VAL_SPLIT_PATH)
    test_paths = load_split(TEST_SPLIT_PATH)
    return train_paths, val_paths, test_paths


@dataclass(frozen=True)
class DataSplit:
    train_paths: list[Path]
    val_paths: list[Path]
    test_paths: list[Path]

    def get_splits(self) -> tuple[list[Path], list[Path], list[Path]]:
        return self.train_paths, self.val_paths, self.test_paths


def create_datasplit() -> DataSplit:
    return DataSplit(*split_parsed_paths())
