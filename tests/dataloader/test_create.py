"""Benchmark test for dataloader creation and iteration speed."""

import time

from dataloader.create import create_dataloaders
from ml_model.id import ModelId
from schemas.batch import Batch
from utils.distributed_context import distributed_context


def test_dataloader_performance():
    """Benchmark dataloader with different num_workers settings."""
    num_workers_list = [0, 4, 16, 64, 256, 1024]
    num_workers_list = [192]  # Default optimal value

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


if __name__ == "__main__":
    test_dataloader_performance()
