import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from constants import PARSED_DIR  # , TEST_SPLIT_PATH, TRAIN_SPLIT_PATH, VAL_SPLIT_PATH
from utils.set_random_seed import set_random_seed


def load_split_filenames(split_file: Path) -> set[str]:
    """Load the exact filenames from a split file."""
    if not split_file.exists():
        return set()
    with open(split_file, "r") as f:
        # Read and strip whitespace, filter empty lines
        return {line.strip() for line in f if line.strip()}


# @lru_cache(maxsize=1)
# def split_parsed_paths() -> tuple[list[Path], list[Path], list[Path]]:
#     """Split parsed paths based on predefined split files with exact filename matching."""
#     train_filenames = load_split_filenames(TRAIN_SPLIT_PATH)
#     val_filenames = load_split_filenames(VAL_SPLIT_PATH)
#     test_filenames = load_split_filenames(TEST_SPLIT_PATH)

#     train_paths = []
#     val_paths = []
#     test_paths = []
#     for parsed_path in PARSED_DIR.glob("*.json"):
#         if parsed_path.stem in train_filenames:
#             train_paths.append(parsed_path)
#         elif parsed_path.stem in val_filenames:
#             val_paths.append(parsed_path)
#         elif parsed_path.stem in test_filenames:
#             test_paths.append(parsed_path)
#         else:
#             raise ValueError(f"Parsed path {parsed_path} not found in any split file")
#     return train_paths, val_paths, test_paths


def split_paths(
    paths: list[Path], train_ratio: float, val_ratio: float
) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled_paths = paths.copy()
    random.shuffle(shuffled_paths)

    train_count = round(len(shuffled_paths) * train_ratio)
    val_count = round(len(shuffled_paths) * val_ratio)

    train_paths = shuffled_paths[:train_count]
    val_paths = shuffled_paths[train_count : train_count + val_count]
    test_paths = shuffled_paths[train_count + val_count :]
    return train_paths, val_paths, test_paths


@lru_cache(maxsize=1)
def split_parsed_paths(
    train_ratio: float = 0.8, val_ratio: float = 0.1
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split parsed paths based on predefined split files with exact filename matching."""
    set_random_seed(42)  # Ensure reproducibility
    parsed_paths = list(PARSED_DIR.glob("*.json"))
    return split_paths(parsed_paths, train_ratio, val_ratio)


@dataclass(frozen=True)
class DataSplit:
    train_paths: list[Path]
    val_paths: list[Path]
    test_paths: list[Path]

    def get_splits(self) -> tuple[list[Path], list[Path], list[Path]]:
        return self.train_paths, self.val_paths, self.test_paths


def create_datasplit() -> DataSplit:
    return DataSplit(*split_parsed_paths())
