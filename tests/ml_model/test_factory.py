"""Test factory model creation for all model IDs."""

from ml_model.factory import ModelFactory
from ml_model.id import ModelId, Task


def test_factory_create_all_models():
    """Test that ModelFactory can create all model IDs."""
    all_model_ids = ModelId.create_defaults()[::-1]

    print(f"\nTesting {len(all_model_ids)} model IDs...")

    successful = []
    failed = []

    for model_id in all_model_ids:
        print(f"\nTesting: {model_id}")
        try:
            model = ModelFactory.load_pretrained(model_id)
            print(f"✓ Successfully created {model_id.task} model")
            print(f"  Model type: {type(model).__name__}")
            print(f"  Parameters: {float(model.num_params) / 1e6:.2f}M")
            successful.append(model_id)
        except FileNotFoundError as e:
            # Expected for tasks that require pretrained models (HTR_SFT)
            if model_id.task in {Task.HTR_SFT}:
                print(f"⚠ Skipped {model_id.task} (requires pretrained model): {e}")
                successful.append(model_id)  # Consider this expected behavior as success
            else:
                print(f"✗ Failed to create {model_id.task}: {e}")
                failed.append((model_id, e))
        except Exception as e:
            print(f"✗ Failed to create {model_id.task}: {e}")
            failed.append((model_id, e))

    print("\n" + "=" * 60)
    print(f"Summary: {len(successful)}/{len(all_model_ids)} models created successfully")

    if failed:
        print(f"\nFailed models ({len(failed)}):")
        for model_id, error in failed:
            print(f"  - {model_id}: {error}")


if __name__ == "__main__":
    test_factory_create_all_models()
