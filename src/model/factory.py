from core.model import ModelFactory, ModelId, Task, LocalModel
from tokeniser.factory import DefaultTokeniserFactory
from model.models.recognition import RecognitionModel
from model.models.generation import GenerationModel
from model.models.pretraining import PretrainingModel
from model.modules.embedder import Embedder, TokenEmbedder, VectorEmbedder
from core.utils.distributed_context import distributed_context


class ReprEmbedderFactory:
    @classmethod
    def create(cls, model_id: ModelId) -> Embedder:
        if model_id.repr_id.is_token:
            if not model_id.repr_id.has_oov:
                unk_token_id = None
            else:
                tokeniser = DefaultTokeniserFactory.create(model_id.repr_id)
                unk_token_id = tokeniser.unk_token_id
            return TokenEmbedder(unk_token_id)

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
                local_model = RecognitionModel(model_id=model_id, repr_embedder=repr_embedder)
            case Task.GENERATION:
                local_model = GenerationModel(model_id=model_id, repr_embedder=repr_embedder)
            case Task.PRETRAINING_NTP:
                local_model = PretrainingModel(model_id=model_id, repr_embedder=repr_embedder)
            case Task.RECOGNITION_SFT:
                local_model = cls._create_recog_sft(model_id)
            case Task.GENERATION_SFT:
                local_model = cls._create_gen_sft(model_id)
            case _:
                raise ValueError(f"Unknown model id: {model_id}")
            
        return local_model.to(distributed_context.device)

    @classmethod
    def _load_pretraining_model(cls, model_id: ModelId) -> PretrainingModel:
        pretrain_model_id = ModelId(task=Task.PRETRAINING_NTP, repr_id=model_id.repr_id)
        pretrain_model = cls.load_pretrained(pretrain_model_id)
        assert isinstance(pretrain_model, PretrainingModel)
        return pretrain_model

    @classmethod
    def _create_recog_sft(cls, model_id: ModelId) -> LocalModel:
        pretrain_model = cls._load_pretraining_model(model_id)
        base_model_id = ModelId(task=Task.RECOGNITION, repr_id=model_id.repr_id)
        local_model = RecognitionModel(model_id=base_model_id,
                                       repr_embedder=pretrain_model.repr_embedder,
                                       decoder=pretrain_model.decoder)
        return local_model

    @classmethod
    def _create_gen_sft(cls, model_id: ModelId) -> LocalModel:
        pretrain_model = cls._load_pretraining_model(model_id)
        base_model_id = ModelId(task=Task.GENERATION, repr_id=model_id.repr_id)
        local_model = GenerationModel(model_id=base_model_id,
                                      repr_embedder=pretrain_model.repr_embedder,
                                      decoder=pretrain_model.decoder)
        return local_model

if __name__ == "__main__":
    for model_id in ModelId.create_defaults():
        model = DefaultModelFactory.create(model_id)
        param_count = model.num_params
        embedding_param_count = model.num_embedding_params
        print(f"Model has {float(param_count)/1e6:.2f}M params (Embedding params: {float(embedding_param_count)/1e6:.2f}M)")