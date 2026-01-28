import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from constants import PARSED_DIR, TEST_SPLIT_PATH, TRAIN_SPLIT_PATH, VAL_SPLIT_PATH
from utils.set_random_seed import set_random_seed


def load_split_filenames(split_file: Path) -> set[str]:
    """Load the exact filenames from a split file."""
    if not split_file.exists():
        return set()
    with open(split_file, "r") as f:
        # Read and strip whitespace, filter empty lines
        return {line.strip() for line in f if line.strip()}


def split_paths(
    split_files: list[Path], train_ratio: float = 0.8, val_ratio: float = 0.1
) -> tuple[list[Path], list[Path], list[Path]]:
    """Randomly split a list of file paths into train, validation, and test sets.

    Args:
        split_files: List of file paths to split
        train_ratio: Proportion of files for training (default: 0.8)
        val_ratio: Proportion of files for validation (default: 0.1)

    Returns:
        Tuple of (train_paths, val_paths, test_paths)
    """
    set_random_seed(seed=42)  # ensure deterministic split

    # Validate ratios
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0 or not (0 <= train_ratio <= 1) or not (0 <= val_ratio <= 1):
        raise ValueError(f"Invalid ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    # Shuffle the files for random split
    shuffled_files = split_files.copy()
    random.shuffle(shuffled_files)

    # Calculate split indices
    total = len(shuffled_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Split the files
    train_paths = shuffled_files[:train_end]
    val_paths = shuffled_files[train_end:val_end]
    test_paths = shuffled_files[val_end:]

    return train_paths, val_paths, test_paths


@lru_cache(maxsize=1)
def split_parsed_paths() -> tuple[list[Path], list[Path], list[Path]]:
    """Split parsed paths based on predefined split files with exact filename matching."""
    train_filenames = load_split_filenames(TRAIN_SPLIT_PATH)
    val_filenames = load_split_filenames(VAL_SPLIT_PATH)
    test_filenames = load_split_filenames(TEST_SPLIT_PATH)

    train_paths = []
    val_paths = []
    test_paths = []
    for parsed_path in PARSED_DIR.glob("*.json"):
        if parsed_path.stem in train_filenames:
            train_paths.append(parsed_path)
        elif parsed_path.stem in val_filenames:
            val_paths.append(parsed_path)
        elif parsed_path.stem in test_filenames:
            test_paths.append(parsed_path)
        else:
            raise ValueError(f"Parsed path {parsed_path} not found in any split file")

    # return split_paths(train_paths + val_paths + test_paths)
    return train_paths, val_paths, test_paths  # TODO:look into this


@dataclass(frozen=True)
class DataSplit:
    train_paths: list[Path]
    val_paths: list[Path]
    test_paths: list[Path]

    def get_splits(self) -> tuple[list[Path], list[Path], list[Path]]:
        return self.train_paths, self.val_paths, self.test_paths


def create_datasplit() -> DataSplit:
    return DataSplit(*split_parsed_paths())
