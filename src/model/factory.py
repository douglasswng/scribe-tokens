from core.model import ModelFactory, ModelId, Task, LocalModel
from core.repr import VectorReprType, TokenReprType
from tokeniser.factory import DefaultTokeniserFactory
from model.models.generation import GenerationModel
from model.modules.embedder import VectorEmbedder, Embedder, TokenEmbedder
from core.utils.distributed_context import distributed_context


class ReprEmbedderFactory:
    @classmethod
    def create(cls, model_id: ModelId) -> Embedder:
        if model_id.task == Task.RECOGNITION:  # uses LSTM
            add_positional_encoding = False
        else:  # uses Transformer
            add_positional_encoding = True

        repr_id = model_id.repr_id

        match repr_id.type:
            case TokenReprType.REL:
                tokeniser = DefaultTokeniserFactory.create(repr_id)
                return TokenEmbedder(unk_token_id=tokeniser.unk_token_id,
                                     token_dropout=UNKNOWN_TOKEN_RATE,
                                     add_positional_encoding=add_positional_encoding)
            case TokenReprType.TEXT | TokenReprType.SCRIBE:
                return TokenEmbedder(unk_token_id=None, token_dropout=0.0,
                                     add_positional_encoding=add_positional_encoding)
            case VectorReprType.POINT3:
                return VectorEmbedder(input_dim=3, add_positional_encoding=add_positional_encoding)
            case VectorReprType.POINT5:
                return VectorEmbedder(input_dim=5, add_positional_encoding=add_positional_encoding)
            case _:
                raise ValueError(f"Invalid embedding type: {repr_id.type}")


class DefaultModelFactory(ModelFactory):
    @classmethod
    def create_local(cls, model_id: ModelId) -> LocalModel:
        repr_embedder = ReprEmbedderFactory.create(model_id)
        match model_id.task:
            case Task.RECOGNITION:
                local_model = RecognitionModel(repr_embedder)
            case Task.GENERATION:
                local_model = GenerationModel(repr_embedder, model_id)
            case _:
                raise ValueError(f"Unknown model id: {model_id}")
            
        return local_model.to(distributed_context.device)
            

if __name__ == "__main__":
    for model_id in ModelId.create_defaults():
        model = DefaultModelFactory.create(model_id)
        print(f"Model has {float(model.num_params)/1e6:.2f}M params")