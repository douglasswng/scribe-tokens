"""Move best checkpoint model weights to models/ directory."""

import torch

from ml_model.id import ModelId


def move_best_checkpoint(model_id: ModelId) -> None:
    """Load best checkpoint and save model weights to models/ directory."""
    best_checkpoint_path = model_id.checkpoint_dir / "best.pt"

    if not best_checkpoint_path.exists():
        print(f"No best checkpoint found for {model_id}")
        return

    state_dict = torch.load(best_checkpoint_path, weights_only=False)
    model_state_dict = state_dict["model_state_dict"]

    model_path = model_id.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state_dict, model_path)
    print(f"Saved model weights to {model_path}")


def main() -> None:
    model_ids = ModelId.create_defaults()
    for model_id in model_ids:
        move_best_checkpoint(model_id)


if __name__ == "__main__":
    main()
