"""Integration test for HTG (Handwriting Generation) model."""

from dataloader.create import create_dataloaders
from ml_model.factory import create_embedder
from ml_model.id import ModelId, Task
from ml_model.locals.htg import HTGModel
from utils.distributed_context import distributed_context


def test_htg_model():
    """Test HTG model forward pass and monitoring."""
    for model_id in ModelId.create_task_model_ids(Task.HTG)[::]:
        print(model_id)
        train_loader, val_loader, test_loader = create_dataloaders(
            model_id=model_id,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        repr_embedder = create_embedder(model_id.repr_id)
        model = HTGModel(repr_embedder=repr_embedder).to(distributed_context.device)

        for batch in train_loader:
            model.train()
            losses = model(batch)
            print(losses)

            model.eval()
            model.monitor(batch)
            break


if __name__ == "__main__":
    test_htg_model()
