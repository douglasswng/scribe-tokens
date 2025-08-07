from pathlib import Path
import random
from dataclasses import dataclass
from functools import lru_cache

from core.utils import set_random_seed
from core.constants import PARSED_DIR, BLACKLIST_PATH


def split_paths(paths: list[Path], train_ratio: float, val_ratio: float
                ) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled_paths = paths.copy()
    random.shuffle(shuffled_paths)

    train_count = int(len(shuffled_paths) * train_ratio)
    val_count = int(len(shuffled_paths) * val_ratio)

    train_paths = shuffled_paths[:train_count]
    val_paths = shuffled_paths[train_count:train_count + val_count]
    test_paths = shuffled_paths[train_count + val_count:]
    return train_paths, val_paths, test_paths


def get_blacklist() -> set[str]:
    with open(BLACKLIST_PATH, 'r') as f:
        return {line.strip() for line in f.readlines()}


@lru_cache(maxsize=1)
def split_parsed_paths() -> tuple[list[Path], list[Path], list[Path]]:
    RANDOM_SEED = 42
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1

    set_random_seed(RANDOM_SEED)
    parsed_paths = list(PARSED_DIR.rglob('*.json'))[:]
    valid_paths = [path for path in parsed_paths if path.stem not in get_blacklist()]
    return split_paths(valid_paths, TRAIN_RATIO, VAL_RATIO)


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