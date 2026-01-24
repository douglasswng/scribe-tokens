"""Integration test for GRPO (Group Relative Policy Optimization) model."""

from dataloader.create import create_dataloaders
from ml_model.factory import ModelFactory
from ml_model.id import ModelId, Task
from utils.distributed_context import distributed_context


def test_grpo_model():
    """Test GRPO model forward pass with reinforcement learning loss."""
    for model_id in ModelId.create_task_model_ids(Task.HTG_GRPO)[:]:
        print(f"Testing GRPO model: {model_id}")
        train_loader, val_loader, test_loader = create_dataloaders(
            model_id=model_id,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        model = ModelFactory.create(model_id).to(distributed_context.device)
        print(f"Model has {float(model.num_params) / 1e6:.2f}M params")

        for batch in train_loader:
            model.train()
            print("Computing GRPO loss...")
            losses = model(batch)
            print(f"Losses: {losses}")

            model.eval()
            print("Monitoring...")
            model.monitor(batch)
            break


if __name__ == "__main__":
    test_grpo_model()
