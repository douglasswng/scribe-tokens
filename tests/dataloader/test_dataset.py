"""Test dataset loading and caching performance."""

from time import time

from dataloader.dataset import create_datasets
from dataloader.split import create_datasplit
from ml_model.id import ModelId


def test_dataset_loading():
    """Test dataset loading speed with and without caching."""
    for _ in range(2):
        for model_id in ModelId.create_defaults():
            print(model_id)
            train_dataset, val_dataset, test_dataset = create_datasets(
                model_id.repr_id, create_datasplit()
            )

            # Test loading speed (should be faster on second access due to caching)
            for _ in range(5):
                start = time()
                train_dataset[0]
                end = time()
                print(f"Time taken: {end - start} seconds")
            print()


if __name__ == "__main__":
    test_dataset_loading()
