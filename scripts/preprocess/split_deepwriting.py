import random
from pathlib import Path

from constants import DATASET, PARSED_DIR, TEST_SPLIT_PATH, TRAIN_SPLIT_PATH, VAL_SPLIT_PATH
from utils.set_random_seed import set_random_seed

assert DATASET == "deepwriting", "Dataset must be deepwriting"


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


def split_parsed_paths(
    train_ratio: float = 0.8, val_ratio: float = 0.1
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split parsed paths based on predefined split files with exact filename matching."""
    set_random_seed(42)  # Ensure reproducibility
    parsed_paths = list(PARSED_DIR.glob("*.json"))
    return split_paths(parsed_paths, train_ratio, val_ratio)


def save_split(paths: list[Path], split_path: Path) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        for path in paths:
            f.write(f"{path.stem}\n")


def main() -> None:
    train_paths, val_paths, test_paths = split_parsed_paths()
    save_split(train_paths, TRAIN_SPLIT_PATH)
    save_split(val_paths, VAL_SPLIT_PATH)
    save_split(test_paths, TEST_SPLIT_PATH)


if __name__ == "__main__":
    main()
