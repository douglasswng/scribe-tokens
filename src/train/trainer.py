from typing import Literal

from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR

from core.model import ModelId, ModelPaths, Model
from core.utils import distributed_context
from model.factory import DefaultModelFactory
from train.tracker import DefaultMLFlowTracker, DefaultSwanLabTracker
from core.train import Trainer, TrainState
from dataloader.create import create_dataloaders
from core.constants import WEIGHT_DECAY, PATIENCE_FACTOR


TRACKER: Literal["mlflow", "swanlab"] = "swanlab"


def get_tracker_class(tracker_type: str):
    tracker_map = {
        "mlflow": DefaultMLFlowTracker,
        "swanlab": DefaultSwanLabTracker,
    }
    return tracker_map[tracker_type]


class DefaultTrainer(Trainer):
    def __init__(self, model_id: ModelId):
        self.model_id = model_id
        self.model_paths = ModelPaths(model_id)

        patience = int(PATIENCE_FACTOR * model_id.task.num_epochs)
        super().__init__(self.model_paths, get_tracker_class(TRACKER)(model_id), patience=patience)

    def _initialise_model(self) -> Model:
        model = DefaultModelFactory.create(self.model_id)
        print(f"Model has {float(model.num_params)/1e6:.2f}M params")
        return model
    
    def _initialise_optimiser(self, model: Model) -> Optimizer:
        return AdamW(model.parameters(), lr=self.model_id.task.learning_rate, weight_decay=WEIGHT_DECAY)

    def _initialise_scheduler(self, optimiser: Optimizer) -> LRScheduler:
        main_scheduler = LambdaLR(optimiser, lr_lambda=lambda epoch: 1.0)
        return main_scheduler
    
    def _initialise_train_state(self) -> TrainState:
        model = self._initialise_model()
        optimiser = self._initialise_optimiser(model)
        scheduler = self._initialise_scheduler(optimiser)
        return TrainState(model, optimiser, scheduler)
    
    def _load_latest_train_state(self, train_state: TrainState) -> TrainState | None:
        latest_train_state = self._checkpointer.load_latest_state(train_state)
        if latest_train_state is None:
            return None
        return latest_train_state

    def train_with_resume(self, num_epochs: int) -> None:
        train_loader, val_loader, _ = create_dataloaders(self.model_id)
        train_state = self._initialise_train_state()
        latest_train_state = self._load_latest_train_state(train_state)
        if latest_train_state is not None:
            if distributed_context.is_master:
                print(f"Found latest checkpoint for {self.model_id}, resuming training")
            train_state = latest_train_state
        else:
            if distributed_context.is_master:
                print(f"No latest checkpoint found for {self.model_id}, starting new training")
        
        distributed_context.barrier()
        self.train(train_state, train_loader, val_loader, num_epochs)
        distributed_context.barrier()