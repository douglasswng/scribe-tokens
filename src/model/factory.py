from core.model import ModelFactory, ModelId, Task, LocalModel
from model.models.recognition import RecognitionModel
from model.modules.embedder import Embedder, TokenEmbedder, VectorEmbedder
from core.utils.distributed_context import distributed_context


class ReprEmbedderFactory:
    @classmethod
    def create(cls, model_id: ModelId) -> Embedder:
        if model_id.repr_id.is_token:
            return TokenEmbedder()

        if model_id.task.is_generation:
            return VectorEmbedder(input_dim=5)
        elif model_id.task.is_recognition:
            return VectorEmbedder(input_dim=3)
        else:
            raise ValueError(f"Unknown model id: {model_id}")


class DefaultModelFactory(ModelFactory):
    @classmethod
    def create_local(cls, model_id: ModelId) -> LocalModel:
        repr_embedder = ReprEmbedderFactory.create(model_id)
        match model_id.task:
            case Task.RECOGNITION:
                local_model = RecognitionModel(repr_embedder)
            case _:
                raise ValueError(f"Unknown model id: {model_id}")
            
        return local_model.to(distributed_context.device)
            

if __name__ == "__main__":
    for model_id in ModelId.create_defaults():
        model = DefaultModelFactory.create(model_id)
        param_count = model.num_params
        embedding_param_count = model.num_embedding_params
        print(f"Model has {float(param_count)/1e6:.2f}M params (Embedding params: {float(embedding_param_count)/1e6:.2f}M)")