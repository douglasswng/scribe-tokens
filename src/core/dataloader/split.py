from pathlib import Path
import random
from dataclasses import dataclass
from functools import lru_cache

from core.model import ModelId
from core.utils import set_random_seed
from core.constants import PARSED_DIR


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


@lru_cache(maxsize=1)
def split_parsed_paths() -> tuple[list[Path], list[Path], list[Path]]:
    RANDOM_SEED = 42  # do not change for reproducibility
    TRAIN_RATIO = 0.8  # do not change for reproducibility
    VAL_RATIO = 0.1  # do not change for reproducibility

    set_random_seed(RANDOM_SEED)
    parsed_paths = list(PARSED_DIR.rglob('*.json'))[:]
    return split_paths(parsed_paths, TRAIN_RATIO, VAL_RATIO)


@dataclass(frozen=True)
class DataSplit:
    train_paths: list[Path]
    val_paths: list[Path]
    test_paths: list[Path]

    def get_splits(self) -> tuple[list[Path], list[Path], list[Path]]:
        return self.train_paths, self.val_paths, self.test_paths


def create_datasplit(model_id: ModelId) -> DataSplit:
    if not model_id.is_sft:
        return DataSplit(*split_parsed_paths())
    
    SFT_TRAIN_RATIO = 0.9
    SFT_VAL_RATIO = 0.1

    _, val_paths, test_paths = split_parsed_paths()  # pretraining data is not used for SFT
    sft_train_paths, sft_val_paths, _ = split_paths(val_paths, SFT_TRAIN_RATIO, SFT_VAL_RATIO)
    return DataSplit(sft_train_paths, sft_val_paths, test_paths)


if __name__ == "__main__":
    for model_id in ModelId.create_defaults():
        split = create_datasplit(model_id)
        print(len(split.train_paths))
        print(len(split.val_paths))
        print(len(split.test_paths))
        print()