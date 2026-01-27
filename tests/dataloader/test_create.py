"""Benchmark test for dataloader creation and iteration speed."""

import time

from dataloader.create import create_dataloaders
from ml_model.id import ModelId
from schemas.batch import Batch
from utils.distributed_context import distributed_context


def test_dataloader_performance():
    """Benchmark dataloader with different num_workers settings."""
    num_workers_list = [0, 4, 16, 64, 256, 1024]

    for model_id in ModelId.create_defaults()[:]:
        if distributed_context.is_master:
            print(f"Model: {model_id}")

        for num_workers in num_workers_list:
            if distributed_context.is_master:
                print(f"  num_workers = {num_workers}")

            train_loader, val_loader, test_loader = create_dataloaders(
                model_id, num_workers=num_workers
            )

            for epoch in range(1):
                start = time.time()
                for batch in train_loader:
                    batch: Batch
                    pass  # Simulate training step
                elapsed = time.time() - start
                if distributed_context.is_master:
                    print(f"    Epoch = {epoch}, Time elapsed: {elapsed:.2f} seconds")


def test_pin_memory_and_persistent_workers():
    """Benchmark dataloader with different pin_memory and persistent_workers settings."""
    # Test configurations: (pin_memory, persistent_workers)
    configs = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    num_workers_list = [64]

    for model_id in ModelId.create_defaults()[:]:
        if distributed_context.is_master:
            print(f"\nModel: {model_id}")

        for num_workers in num_workers_list:
            if distributed_context.is_master:
                print(f"  num_workers = {num_workers}")

            for pin_memory, persistent_workers in configs:
                if distributed_context.is_master:
                    print(f"    pin_memory={pin_memory}, persistent_workers={persistent_workers}")

                train_loader, val_loader, test_loader = create_dataloaders(
                    model_id,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                )

                # Verify the settings are applied correctly
                assert train_loader.pin_memory == pin_memory
                expected_persistent = persistent_workers and num_workers > 0
                assert train_loader.persistent_workers == expected_persistent

                for epoch in range(1):
                    start = time.time()
                    for batch in train_loader:
                        batch: Batch
                        pass  # Simulate training step
                    elapsed = time.time() - start
                    if distributed_context.is_master:
                        print(f"      Epoch = {epoch}, Time elapsed: {elapsed:.2f} seconds")


if __name__ == "__main__":
    test_pin_memory_and_persistent_workers()
    test_dataloader_performance()
