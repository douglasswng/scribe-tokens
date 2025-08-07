from pathlib import Path

from core.model.id import ModelId
from core.constants import CHECKPOINTS_DIR, MODELS_DIR, METRICS_DIR


class ModelPaths:
    def __init__(self, model_id: ModelId):
        self._model_id = model_id
    
    @property
    def _base_dir(self) -> Path:
        return Path(self._model_id.task.value) / self._model_id.repr_id.type.value
    
    @property
    def checkpoint_dir(self) -> Path:
        return CHECKPOINTS_DIR / self._base_dir
    
    @property
    def model_path(self) -> Path:
        return MODELS_DIR / self._base_dir / 'best.pt'
    
    @property
    def metrics_path(self) -> Path:
        return METRICS_DIR / self._base_dir / 'metrics.json'
    
    def list_checkpoint_paths(self) -> list[Path]:
        return list(self.checkpoint_dir.glob("checkpoint_*.pt"))
    
    def get_latest_checkpoint_path(self) -> Path | None:
        checkpoint_paths = self.list_checkpoint_paths()
        if not checkpoint_paths:
            return None
        
        latest_path = max(checkpoint_paths, key=lambda p: int(p.stem.split('_')[1]))
        return latest_path
    
    def get_checkpoint_path(self, epoch: int) -> Path:
        return self.checkpoint_dir / f'checkpoint_{epoch}.pt'
    

if __name__ == '__main__':
    for model_id in ModelId.create_defaults():
        model_paths = ModelPaths(model_id)
        print(model_paths.checkpoint_dir)
        print(model_paths.model_path)
        print(model_paths.list_checkpoint_paths())
        print(model_paths.get_checkpoint_path(10))