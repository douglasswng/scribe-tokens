from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache

from core.constants import PARSED_DIR, SPLIT_DIR


def load_split_filenames(split_file: Path) -> set[str]:
    """Load the exact filenames from a split file."""
    if not split_file.exists():
        return set()
    with open(split_file, 'r') as f:
        # Read and strip whitespace, filter empty lines
        return {line.strip() for line in f if line.strip()}


@lru_cache(maxsize=1)
def split_parsed_paths() -> tuple[list[Path], list[Path], list[Path]]:
    """Split parsed paths based on predefined split files with exact filename matching."""
    # Find the dataset directory (assumes there's one subdirectory in parsed/)
    dataset_dirs = [d for d in PARSED_DIR.iterdir() if d.is_dir()]
    if not dataset_dirs:
        raise ValueError(f"No dataset directory found in {PARSED_DIR}")
    
    dataset_dir = dataset_dirs[0]  # Use the first dataset directory
    dataset_name = dataset_dir.name
    split_dataset_dir = SPLIT_DIR / dataset_name
    
    # Load exact filenames from split files
    train_filenames = load_split_filenames(split_dataset_dir / 'train.txt')
    test_filenames = load_split_filenames(split_dataset_dir / 'test.txt')
    
    # Combine val1 and val2
    val_filenames = load_split_filenames(split_dataset_dir / 'val1.txt')
    val_filenames.update(load_split_filenames(split_dataset_dir / 'val2.txt'))
    
    # Build paths using exact filename matching
    train_paths = sorted([dataset_dir / fname for fname in train_filenames if (dataset_dir / fname).exists()])
    val_paths = sorted([dataset_dir / fname for fname in val_filenames if (dataset_dir / fname).exists()])
    test_paths = sorted([dataset_dir / fname for fname in test_filenames if (dataset_dir / fname).exists()])
    
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


if __name__ == "__main__":
    split = create_datasplit()
    print(len(split.train_paths))
    print(len(split.val_paths))
    print(len(split.test_paths))
    print()