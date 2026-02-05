import torch

from ink_repr.id import VectorReprId
from ink_tokeniser.factory import TokeniserFactory
from ink_tokeniser.id import TokeniserId
from ml_model.id import ModelId, Task
from ml_model.locals.htg import HTGModel
from ml_model.locals.htr import HTRModel
from ml_model.locals.local import LocalModel
from ml_model.locals.ntp import NTPModel
from ml_model.model import DDPModel, Model
from ml_model.modules.embedder import Embedder, TokenEmbedder, VectorEmbedder
from utils.distributed_context import distributed_context


def create_embedder(model_id: ModelId) -> Embedder:
    match model_id.repr_id:
        case TokeniserId():
            tokeniser = TokeniserFactory.create(model_id.repr_id)
            return TokenEmbedder(unk_token_id=tokeniser.unk_token_id)
        case VectorReprId():
            # HTR tasks only need embedding, not unembedding (no MDN output layers)
            needs_unembed = model_id.task in {Task.HTG, Task.NTP, Task.HTG_SFT}
            return VectorEmbedder(model_id.repr_id.dim, with_unembed=needs_unembed)
        case _:
            raise ValueError(f"Unsupported repr id: {model_id.repr_id}")


class ModelFactory:
    @classmethod
    def _create_local(cls, model_id: ModelId) -> LocalModel:
        embedder = create_embedder(model_id)
        match model_id.task:
            case Task.HTR:
                model = HTRModel(repr_embedder=embedder)
            case Task.HTG:
                model = HTGModel(repr_embedder=embedder)
            case Task.NTP:
                model = NTPModel(repr_embedder=embedder)
            case Task.HTR_SFT:
                ntp_model_id = ModelId(task=Task.NTP, repr_id=model_id.repr_id)
                ntp_model = cls.load_pretrained(ntp_model_id).local_model
                assert isinstance(ntp_model, NTPModel)
                model = HTRModel(repr_embedder=ntp_model.repr_embedder, decoder=ntp_model.decoder)
            case Task.HTG_SFT:
                ntp_model_id = ModelId(task=Task.NTP, repr_id=model_id.repr_id)
                ntp_model = cls.load_pretrained(ntp_model_id).local_model
                assert isinstance(ntp_model, NTPModel)
                model = HTGModel(repr_embedder=ntp_model.repr_embedder, decoder=ntp_model.decoder)
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
        state_dict = torch.load(model_id.model_path, weights_only=False, map_location="cpu")
        model.load_state_dict(state_dict)
        if distributed_context.is_distributed:
            model = DDPModel(model)
        return model
