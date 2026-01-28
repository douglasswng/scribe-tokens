"""Test that all pretrained models can be loaded and run monitor."""

import torch

from dataloader.create import create_dataloaders
from ml_model.factory import ModelFactory
from ml_model.id import ModelId
from schemas.batch import Batch
from schemas.instance import Instance
from utils.distributed_context import distributed_context


def move_batch_to_device(batch: Batch) -> Batch:
    """Move all tensors in the batch to the appropriate device."""
    device = distributed_context.device
    moved_instances = [
        Instance(
            parsed=instance.parsed,
            repr_id=instance.repr_id,
            repr=instance.repr.to(device, non_blocking=True),
            char=instance.char.to(device, non_blocking=True),
        )
        for instance in batch.instances
    ]
    return Batch(instances=moved_instances)


def test_pretrained_models_monitor():
    """Test that all pretrained models can be loaded and run monitor."""
    all_model_ids = ModelId.create_defaults()

    print(f"\nTesting monitor for {len(all_model_ids)} model IDs...")

    successful = []
    skipped = []
    failed = []

    for model_id in all_model_ids:
        if model_id.repr_id.type not in {"Point-5", "RelTokens"}:
            continue
        print(f"\nTesting: {model_id}")
        try:
            model = ModelFactory.load_pretrained(model_id)
            print(f"  Loaded {type(model.local_model).__name__} ({model.num_params / 1e6:.2f}M)")

            train_loader, _, _ = create_dataloaders(
                model_id=model_id,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )

            model.eval()
            with torch.no_grad():
                for batch in train_loader:
                    batch = move_batch_to_device(batch)
                    model.monitor(batch)
                    break

            print("  Monitor completed successfully")
            successful.append(model_id)

        except FileNotFoundError as e:
            print(f"  Skipped (no pretrained model): {e}")
            skipped.append(model_id)
        except Exception as e:
            print(f"  Failed: {e}")
            failed.append((model_id, e))

    print("\n" + "=" * 60)
    print(f"Summary: {len(successful)} successful, {len(skipped)} skipped, {len(failed)} failed")

    if failed:
        print(f"\nFailed models ({len(failed)}):")
        for model_id, error in failed:
            print(f"  - {model_id}: {error}")
        raise AssertionError(f"{len(failed)} models failed to run monitor")


if __name__ == "__main__":
    test_pretrained_models_monitor()
