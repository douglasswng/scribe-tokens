from pathlib import Path
from collections import defaultdict
import random
from functools import partial

from torch.utils.data import Dataset

from core.data_schema import Parsed, Instance
from core.model import ModelId
from repr.factory import DefaultReprFactory
from dataloader.augmenter import Augmenter
from dataloader.split import DataSplit


class ParsedDataset(Dataset):
    def __init__(self,
                 model_id: ModelId,
                 parsed_paths: list[Path],
                 augment: bool):
        self._parsed_paths = parsed_paths
        self._augment = augment

        self._repr_callable = partial(DefaultReprFactory.ink_to_tensor, model_id.repr_id)
        self._instance_callable = partial(Instance, context_type=model_id.context_type, main_type=model_id.main_type)
        self._writer_idxs = self._build_writer_idxs()  # for reference
        self._parsed_to_instance: dict[str, Instance] = {}  # for caching

    def _build_writer_idxs(self) -> defaultdict[str, set[int]]:
        writer_idxs = defaultdict[str, set[int]](set)
        for idx, parsed_path in enumerate(self._parsed_paths):
            parsed = Parsed.from_path(parsed_path)
            writer = parsed.writer
            writer_idxs[writer].add(idx)
        return writer_idxs

    def __len__(self) -> int:
        return len(self._parsed_paths)
    
    def _get_parsed(self, idx: int) -> Parsed:
        parsed_path = self._parsed_paths[idx]
        parsed = Parsed.from_path(parsed_path)
        if self._augment:
            parsed = Augmenter.augment(parsed)
        return parsed
    
    def _to_instance(self, parsed: Parsed) -> Instance:
        repr = self._repr_callable(parsed.ink)
        return Instance(parsed=parsed, _repr=repr)
    
    def _get_instance(self, parsed: Parsed) -> Instance:
        if self._augment:
            return self._to_instance(parsed)
        
        if parsed.id not in self._parsed_to_instance:
            instance = self._to_instance(parsed)
            self._parsed_to_instance[parsed.id] = instance
        return self._parsed_to_instance[parsed.id]
    
    def __getitem__(self, idx: int) -> Instance:
        if self._augment:
            Augmenter.reset_config()

        parsed = self._get_parsed(idx)
        instance = self._get_instance(parsed)
        return instance
    

def create_datasets(model_id: ModelId, datasplit: DataSplit
                    ) -> tuple[ParsedDataset, ParsedDataset, ParsedDataset]:
    partial_parsed_dataset = partial(ParsedDataset,
                                     model_id=model_id)
    
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
                main_instance, reference_instance = train_dataset[0]
                if reference_instance is not None:
                    main_instance.parsed.visualise()
                    reference_instance.parsed.visualise()
                end = time()
                print(f"Time taken: {end - start} seconds")
            print()