from functools import partial
from pathlib import Path

from model.id import ModelId
from torch.utils.data import Dataset

from dataloader.augmenter import Augmenter
from dataloader.split import DataSplit
from ink_repr.factory import ReprFactory
from schemas.instance import Instance
from schemas.parsed import Parsed


class ParsedDataset(Dataset):
    def __init__(self, model_id: ModelId, parsed_paths: list[Path], augment: bool):
        self._model_id = model_id
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

        repr = ReprFactory.from_ink(parsed.ink, repr_id=self._model_id.repr_id)
        return Instance(parsed=parsed, _repr_tensor=repr.to_tensor())

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


def create_datasets(
    model_id: ModelId, datasplit: DataSplit
) -> tuple[ParsedDataset, ParsedDataset, ParsedDataset]:
    partial_parsed_dataset = partial(ParsedDataset, model_id=model_id)

    train_paths, val_paths, test_paths = datasplit.get_splits()
    train_dataset = partial_parsed_dataset(parsed_paths=train_paths, augment=True)
    val_dataset = partial_parsed_dataset(parsed_paths=val_paths, augment=False)
    test_dataset = partial_parsed_dataset(parsed_paths=test_paths, augment=False)
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    from time import time

    from dataloader.split import create_datasplit

    for _ in range(2):
        for model_id in ModelId.create_defaults():
            print(model_id)
            train_dataset, val_dataset, test_dataset = create_datasets(model_id, create_datasplit())
            for _ in range(5):
                start = time()
                instance = train_dataset[0]
                end = time()
                print(f"Time taken: {end - start} seconds")
            print()
