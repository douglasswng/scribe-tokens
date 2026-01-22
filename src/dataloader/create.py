from functools import partial
from typing import cast

from core.constants import BATCH_SIZE
from core.utils import distributed_context
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from core.data_schema import Batch, Instance, PairBatch, SingletonBatch
from core.model import ModelId
from dataloader.dataset import create_datasets
from dataloader.split import create_datasplit


def collate_fn(instances: list[Instance | tuple[Instance, Instance]]) -> Batch:
    first_instance = instances[0]
    is_singleton_batch = isinstance(first_instance, Instance)

    if not all(isinstance(instance, type(first_instance)) for instance in instances):
        raise ValueError(
            "Mixed instance types in batch - all instances must be either Instance or tuple[Instance, Instance]"
        )

    if is_singleton_batch:
        return SingletonBatch(instances=cast(list[Instance], instances))
    else:
        tuple_instances = cast(list[tuple[Instance, Instance]], instances)
        main_instances, ref_instances = zip(*tuple_instances)
        return PairBatch(main_instances=list(main_instances), ref_instances=list(ref_instances))


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    if distributed_context.is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed_context.world_size,
            rank=distributed_context.rank,
            shuffle=shuffle,
        )
        shuffle = False
        drop_last = True  # uneven batchsizes causes mean of means issues
    else:
        sampler = None
        drop_last = False

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )


def create_dataloaders(
    model_id: ModelId,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 192,  # seems decently optimal, same as number of cores
    pin_memory: bool = True,
    persistent_workers: bool = True,  # seems to speed up
) -> tuple[DataLoader, DataLoader, DataLoader]:
    datasplit = create_datasplit()
    device_batch_size = distributed_context.divide_by_world_size(batch_size)
    device_num_workers = distributed_context.divide_by_world_size(num_workers)
    persistent_workers = persistent_workers and device_num_workers > 0
    partial_create_dataloader = partial(
        create_dataloader,
        batch_size=device_batch_size,
        num_workers=device_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    train_dataset, val_dataset, test_dataset = create_datasets(model_id, datasplit)
    train_dataloader = partial_create_dataloader(dataset=train_dataset, shuffle=True)
    val_dataloader = partial_create_dataloader(dataset=val_dataset, shuffle=False)
    test_dataloader = partial_create_dataloader(dataset=test_dataset, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    import time

    num_workers_list = [0, 4, 16, 64, 256, 1024]
    num_workers_list = [192]
    for model_id in ModelId.create_defaults()[:]:
        if distributed_context.is_master:
            print(f"Model: {model_id}")
        for num_workers in num_workers_list:
            if distributed_context.is_master:
                print(f"  num_workers = {num_workers}")
            train_loader, val_loader, test_loader = create_dataloaders(
                model_id, num_workers=num_workers
            )

            for epoch in range(2):
                start = time.time()
                for batch in train_loader:
                    batch: Batch
                    pass  # Simulate training step
                elapsed = time.time() - start
                if distributed_context.is_master:
                    print(f"    Epoch = {epoch}, Time elapsed: {elapsed:.2f} seconds")
