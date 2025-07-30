from typing import Protocol

import torch

from core.model.id import ModelId
from core.model.model import LocalModel, DistributedModel, Model
from core.model.paths import ModelPaths
from core.utils.distributed_context import distributed_context


class ModelFactory(Protocol):
    @classmethod
    def create_local(cls, model_id: ModelId) -> LocalModel: ...

    @classmethod
    def create(cls, model_id: ModelId) -> Model:
        model = cls.create_local(model_id)
        model.init_weights()
        if distributed_context.is_distributed:
            model = DistributedModel(model)
        return model
    
    @classmethod
    def load_pretrained(cls, model_id: ModelId) -> Model:
        model = cls.create_local(model_id)
        model_paths = ModelPaths(model_id)
        state_dict = torch.load(model_paths.model_path,
                                weights_only=True,
                                map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        if distributed_context.is_distributed:
            model = DistributedModel(model)
        return model