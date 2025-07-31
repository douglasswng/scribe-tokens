from core.model import ModelFactory, ModelId, Task, LocalModel
from model.models.generation import GenerationModel
from core.utils.distributed_context import distributed_context


class DefaultModelFactory(ModelFactory):
    @classmethod
    def create_local(cls, model_id: ModelId) -> LocalModel:
        match model_id.task:
            case Task.GENERATION:
                local_model = GenerationModel(model_id)
            case _:
                raise ValueError(f"Unknown model id: {model_id}")
            
        return local_model.to(distributed_context.device)
            

if __name__ == "__main__":
    for model_id in ModelId.create_defaults():
        model = DefaultModelFactory.create(model_id)
        param_count = model.num_params
        embedding_param_count = model.num_embedding_params
        print(f"Model has {float(param_count)/1e6:.2f}M params (Embedding params: {float(embedding_param_count)/1e6:.2f}M)")