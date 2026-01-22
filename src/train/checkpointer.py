from ml_model.id import ModelId
from train.state import TrainState


class Checkpointer:
    def __init__(self, model_id: ModelId, checkpoint_interval: int = 20, max_checkpoints: int = 10):
        self._model_id = model_id
        self._checkpoint_interval = checkpoint_interval
        self._max_checkpoints = max_checkpoints

    def _prune_checkpoints(self) -> None:
        checkpoint_paths = self._model_id.list_checkpoint_paths()
        if len(checkpoint_paths) <= self._max_checkpoints:
            return

        checkpoint_paths.sort(key=lambda p: int(p.stem.split("_")[1]))
        for path in checkpoint_paths[: -self._max_checkpoints]:
            path.unlink()

    def save_model(self, train_state: TrainState) -> None:
        model_path = self._model_id.model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        train_state.save_model(model_path)

    def save_state(self, train_state: TrainState) -> None:
        checkpoint_path = self._model_id.get_checkpoint_path(train_state.epoch)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        train_state.save_state(checkpoint_path)
        self._prune_checkpoints()

    def should_save_state(self, train_state: TrainState) -> bool:
        return (train_state.epoch - 1) % self._checkpoint_interval == 0

    def load_state(self, epoch: int, train_state: TrainState) -> TrainState:
        checkpoint_path = self._model_id.get_checkpoint_path(epoch)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
        return train_state.load_state(checkpoint_path)

    def load_latest_state(self, train_state: TrainState) -> TrainState | None:
        latest_path = self._model_id.get_latest_checkpoint_path()
        if latest_path is None:
            return None
        return train_state.load_state(latest_path)
