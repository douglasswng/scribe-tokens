import random
from collections import defaultdict
from functools import partial
from pathlib import Path

from torch.utils.data import Dataset

from core.data_schema import Instance, Parsed
from core.model import ModelId
from dataloader.augmenter import Augmenter
from dataloader.split import DataSplit
from repr.factory import DefaultReprFactory


class ParsedDataset(Dataset):
    def __init__(self, model_id: ModelId, parsed_paths: list[Path], augment: bool):
        self._model_id = model_id
        self._parsed_paths = parsed_paths
        self._augment = augment

        self._repr_callable = partial(DefaultReprFactory.ink_to_tensor, model_id.repr_id)
        self._parsed_to_instance: dict[str, Instance] = {}  # for caching when no need to augment
        if self._model_id.task.use_reference:
            self._writer_idxs = self._build_writer_idxs()

    def __len__(self) -> int:
        return len(self._parsed_paths)

    def _build_writer_idxs(self) -> defaultdict[str, set[int]]:
        writer_idxs = defaultdict[str, set[int]](set)
        for idx, parsed_path in enumerate(self._parsed_paths):
            parsed = Parsed.from_path(parsed_path)
            writer = parsed.writer
            writer_idxs[writer].add(idx)
        return writer_idxs

    def _get_parsed(self, idx: int) -> Parsed:
        parsed_path = self._parsed_paths[idx]
        parsed = Parsed.from_path(parsed_path)
        if self._augment:
            parsed = Augmenter.augment(parsed)
        return parsed

    def _to_instance(self, parsed: Parsed) -> Instance:
        repr = self._repr_callable(parsed.ink)
        return Instance(parsed=parsed, _repr_tensor=repr)

    def _get_instance(self, parsed: Parsed) -> Instance:
        if self._augment:
            return self._to_instance(parsed)

        if parsed.id not in self._parsed_to_instance:
            instance = self._to_instance(parsed)
            self._parsed_to_instance[parsed.id] = instance
        return self._parsed_to_instance[parsed.id]

    def __getitem__(self, idx: int) -> Instance | tuple[Instance, Instance]:
        if self._augment:
            Augmenter.reset_config()

        main_parsed = self._get_parsed(idx)
        main_instance = self._get_instance(main_parsed)
        if not self._model_id.task.use_reference:
            return main_instance

        valid_ref_idxs = self._writer_idxs[main_parsed.writer] - {idx}
        valid_ref_list = list(valid_ref_idxs)
        ref_idx = random.choice(valid_ref_list) if valid_ref_list else idx
        ref_parsed = self._get_parsed(ref_idx)
        ref_instance = self._get_instance(ref_parsed)
        return main_instance, ref_instance


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
