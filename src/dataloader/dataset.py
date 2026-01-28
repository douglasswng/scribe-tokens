import random
from functools import partial
from pathlib import Path

from english_words import get_english_words_set
from torch.utils.data import Dataset

from dataloader.augmenter import Augmenter
from dataloader.split import DataSplit
from ink_repr.id import ReprId
from schemas.instance import Instance
from schemas.parsed import Parsed


class ParsedDataset(Dataset):
    def __init__(self, repr_id: ReprId, parsed_paths: list[Path], augment: bool):
        self._repr_id = repr_id
        self._parsed_paths = parsed_paths
        self._augment = augment

        self._cache: dict[int, Instance] = {}

    def __len__(self) -> int:
        return len(self._parsed_paths)

    def _get_instance(self, idx: int) -> Instance:
        """Helper to do the heavy lifting."""
        if self._augment:
            Augmenter.reset_config()

        parsed_path = self._parsed_paths[idx]
        parsed = Parsed.from_path(parsed_path)

        if self._augment:
            parsed = Augmenter.augment(parsed)

        return Instance.from_parsed(parsed, repr_id=self._repr_id)

    def __getitem__(self, idx: int) -> Instance:
        # Case 1: Augmentation is on. Never cache.
        if self._augment:
            return self._get_instance(idx)

        # Case 2: Augmentation is off. Check cache.
        if idx in self._cache:
            return self._cache[idx]

        # Case 3: Not in cache. Process and store.
        instance = self._get_instance(idx)
        self._cache[idx] = instance
        return instance


class WordDataset(Dataset):
    def __init__(self, repr_id: ReprId, len: int):
        self._repr_id = repr_id
        self._len = len
        self._words = list(get_english_words_set(("web2",), alpha=True))

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Instance:
        word = random.choice(self._words)
        return Instance.from_text(word, self._repr_id)


def create_datasets(
    repr_id: ReprId, datasplit: DataSplit
) -> tuple[ParsedDataset, ParsedDataset, ParsedDataset]:
    partial_parsed_dataset = partial(ParsedDataset, repr_id=repr_id)

    train_paths, val_paths, test_paths = datasplit.get_splits()
    train_dataset = partial_parsed_dataset(parsed_paths=train_paths, augment=True)
    val_dataset = partial_parsed_dataset(parsed_paths=val_paths, augment=False)
    test_dataset = partial_parsed_dataset(parsed_paths=test_paths, augment=False)
    return train_dataset, val_dataset, test_dataset
