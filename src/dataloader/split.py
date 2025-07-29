from pathlib import Path
import random
from dataclasses import dataclass
from functools import lru_cache

from core.utils import set_random_seed
from core.constants import PARSED_DIR


def split_paths(paths: list[Path], train_ratio: float) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled_paths = paths.copy()
    random.shuffle(shuffled_paths)

    train_count = int(len(shuffled_paths) * train_ratio)

    train_paths = shuffled_paths[:train_count]
    val_paths = shuffled_paths[train_count:]
    test_paths = []  # not need test set

    return train_paths, val_paths, test_paths


@lru_cache(maxsize=1)
def split_parsed_paths() -> tuple[list[Path], list[Path], list[Path]]:
    RANDOM_SEED = 42
    TRAIN_RATIO = 0.95

    set_random_seed(RANDOM_SEED)
    parsed_paths = list(PARSED_DIR.rglob('*.json'))[:]
    return split_paths(parsed_paths, TRAIN_RATIO)


@dataclass(frozen=True)
class DataSplit:
    train_paths: list[Path]
    val_paths: list[Path]
    test_paths: list[Path]

    def get_splits(self) -> tuple[list[Path], list[Path], list[Path]]:
        return self.train_paths, self.val_paths, self.test_paths


def create_datasplit() -> DataSplit:
    return DataSplit(*split_parsed_paths())


if __name__ == "__main__":
    split = create_datasplit()
    print(len(split.train_paths))
    print(len(split.val_paths))
    print(len(split.test_paths))
    print()