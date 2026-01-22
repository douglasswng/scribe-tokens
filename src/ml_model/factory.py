from ml_model.id import ModelId
from ml_model.model import Model


class ModelFactory:
    @classmethod
    def create(cls, model_id: ModelId) -> Model: ...
