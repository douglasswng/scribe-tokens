from functools import partial

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from constants import BATCH_SIZE
from dataloader.dataset import create_datasets
from dataloader.split import create_datasplit
from ml_model.id import ModelId
from schemas.batch import Batch
from utils.distributed_context import distributed_context


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
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
        collate_fn=lambda x: Batch(instances=x),
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        prefetch_factor=4,
    )


def create_dataloaders(
    model_id: ModelId,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 64,  # seems decently optimal, same as number of cores
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
