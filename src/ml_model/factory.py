import torch

from ink_repr.id import ReprId, VectorReprId
from ink_tokeniser.factory import TokeniserFactory
from ink_tokeniser.id import TokeniserId
from ml_model.id import ModelId, Task
from ml_model.locals.grpo import GRPOModel
from ml_model.locals.htg import HTGModel
from ml_model.locals.htr import HTRModel
from ml_model.locals.local import LocalModel
from ml_model.locals.ntp import NTPModel
from ml_model.model import DDPModel, Model
from ml_model.modules.embedder import Embedder, TokenEmbedder, VectorEmbedder
from utils.distributed_context import distributed_context


def create_embedder(repr_id: ReprId) -> Embedder:
    match repr_id:
        case TokeniserId():
            tokeniser = TokeniserFactory.create(repr_id)
            return TokenEmbedder(unk_token_id=tokeniser.unk_token_id)
        case VectorReprId():
            return VectorEmbedder(repr_id.dim)
        case _:
            raise ValueError(f"Unsupported repr id: {repr_id}")


class ModelFactory:
    @classmethod
    def _create_local(cls, model_id: ModelId) -> LocalModel:
        embedder = create_embedder(model_id.repr_id)
        match model_id.task:
            case Task.HTR:
                model = HTRModel(repr_embedder=embedder)
            case Task.HTG:
                model = HTGModel(repr_embedder=embedder)
            case Task.NTP:
                model = NTPModel(repr_embedder=embedder)
            case Task.HTR_SFT:
                ntp_model_id = ModelId(task=Task.NTP, repr_id=model_id.repr_id)
                ntp_model = cls.load_pretrained(ntp_model_id)
                assert isinstance(ntp_model, NTPModel)
                model = HTRModel(repr_embedder=ntp_model.repr_embedder, decoder=ntp_model.decoder)
            case Task.HTG_GRPO:
                htg_model_id = ModelId(task=Task.HTG, repr_id=model_id.repr_id)
                htg_model = cls.load_pretrained(htg_model_id)
                assert isinstance(htg_model, HTGModel)

                htr_model_id = ModelId(task=Task.HTR_SFT, repr_id=model_id.repr_id)  # SFT better
                htr_model = cls.load_pretrained(htr_model_id)
                assert isinstance(htr_model, HTRModel)

                model = GRPOModel(htg_model=htg_model, htr_model=htr_model)
            case _:
                raise ValueError(f"Unsupported task: {model_id.task}")
        if model_id.task.need_init_weights:
            model.init_weights()
        return model.to(distributed_context.device)

    @classmethod
    def create(cls, model_id: ModelId) -> Model:
        model = cls._create_local(model_id)
        if distributed_context.is_distributed:
            model = DDPModel(model)
        return model

    @classmethod
    def load_pretrained(cls, model_id: ModelId) -> Model:
        model = cls._create_local(model_id)
        state_dict = torch.load(model_id.model_path, weights_only=True, map_location="cpu")
        model.load_state_dict(state_dict)
        if distributed_context.is_distributed:
            model = DDPModel(model)
        return model


if __name__ == "__main__":
    for model_id in ModelId.create_defaults():
        model = ModelFactory.create(model_id)
        print(f"Model ({model_id}) has {float(model.num_params) / 1e6:.2f}M params")
