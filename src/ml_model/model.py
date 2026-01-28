from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from ml_model.locals.local import LocalModel
from ml_trainer.tracker import Tracker
from schemas.batch import Batch
from utils.distributed_context import distributed_context


class DDPModel(DDP):
    def __init__(self, local_model: LocalModel):
        if not distributed_context.is_distributed:
            raise ValueError("DistributedModel can only be used in distributed mode")

        super().__init__(local_model, device_ids=distributed_context.device_ids)

    @property
    def local_model(self) -> LocalModel:
        return self.module

    def monitor(self, batch: Batch) -> None:
        return self.local_model.monitor(batch)

    def validation_losses(self, batch: Batch) -> dict[str, Tensor]:
        return self.local_model.validation_losses(batch)

    def set_tracker(self, tracker: Tracker) -> None:
        self.local_model.set_tracker(tracker)

    @property
    def num_params(self) -> int:
        return self.local_model.num_params


type Model = LocalModel | DDPModel
