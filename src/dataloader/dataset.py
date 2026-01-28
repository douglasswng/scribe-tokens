from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Dataset

from constants import CHARS, NUM_CHARS
from dataloader.augmenter import Augmenter
from dataloader.split import DataSplit
from ink_repr.factory import ReprFactory
from ml_model.id import ModelId
from schemas.instance import Instance
from schemas.parsed import Parsed


class IdMapper:
    _CHAR_ID_MAP = {char: id for id, char in enumerate(CHARS, 1)}
    _ID_CHAR_MAP = {v: k for k, v in _CHAR_ID_MAP.items()}

    @classmethod
    def chars_to_ids(cls, chars: list[str]) -> list[int]:
        return [cls._CHAR_ID_MAP[char] for char in chars]

    @classmethod
    def ids_to_chars(cls, ids: list[int]) -> list[str]:
        return [cls._ID_CHAR_MAP.get(id, "") for id in ids]  # empty string for bos and eos

    @classmethod
    def str_to_ids(cls, s: str) -> list[int]:
        bos_id = NUM_CHARS + 1
        eos_id = NUM_CHARS + 2

        ids = cls.chars_to_ids(list(s))
        ids = [bos_id] + ids + [eos_id]
        return ids

    @classmethod
    def ids_to_str(cls, ids: list[int]) -> str:
        return "".join(cls.ids_to_chars(ids))


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
        repr_tensor = repr.to_tensor()
        char_tensor = torch.tensor(IdMapper.str_to_ids(parsed.text))

        return Instance(
            parsed=parsed, repr_id=self._model_id.repr_id, repr=repr_tensor, char=char_tensor
        )

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
