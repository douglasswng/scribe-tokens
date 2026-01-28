import torch
from torch import Tensor

from ink_repr.factory import ReprFactory
from ml_model.locals.local import ForwardOutput, KVCaches, LocalModel
from ml_model.modules.decoder import TransformerDecoder
from ml_model.modules.embedder import Embedder
from schemas.batch import Batch
from schemas.ink import DigitalInk
from schemas.instance import Instance


class NTPModel(LocalModel):
    """Next Token Prediction model that predicts next repr given previous repr."""

    def __init__(self, repr_embedder: Embedder):
        super().__init__()
        self.repr_embedder = repr_embedder
        self.decoder = TransformerDecoder()

    def _losses(self, batch: Batch) -> dict[str, Tensor]:
        input, target, loss_mask = self._prepare_batch(
            batch=batch,
            prompt_embedder=None,
            completion_embedder=self.repr_embedder,
            prompt_attr=None,
            completion_input_attr="repr_input",
            completion_target_attr="repr_target",
        )

        pred = self._forward(input)
        match pred:
            case Tensor():
                return {"ce": self.ce_loss(pred, target, loss_mask)}
            case tuple() if len(pred) == 5:
                return {"nll": self.nll_loss(pred, target, loss_mask)}
            case _:
                raise ValueError(f"Unsupported prediction type: {type(pred)}")

    def _forward(
        self,
        input: Tensor,
        start_pos: int | Tensor = 0,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        attn_mask: Tensor | None = None,
        use_cache: bool = False,
    ) -> ForwardOutput | tuple[ForwardOutput, KVCaches]:
        result = self.decoder(
            input,
            start_pos=start_pos,
            kv_caches=kv_caches,
            attn_mask=attn_mask,
            use_cache=use_cache,
        )
        if not use_cache:
            assert isinstance(result, Tensor)
            pred = self.repr_embedder.unembed(result)
            return pred
        else:
            assert isinstance(result, tuple)
            output, new_kv_caches = result
            pred = self.repr_embedder.unembed(output)
            return pred, new_kv_caches

    def monitor(self, batch: Batch) -> None:
        instance = batch.get_random_instance()
        ink_pred = self.generate_inks(instance)[0]
        self._track_ink(ink=ink_pred, task="NTP")

    @torch.no_grad()
    def generate_inks(
        self,
        instance: Instance,
        max_len: int = 500,
        temperature: float = 1.0,
        num_generations: int = 1,
    ) -> list[DigitalInk]:
        gens = self._generate_sequences(
            context=[None],
            bos=instance.repr_bos,
            eos=instance.repr_eos,
            output_embedder=self.repr_embedder,
            max_len=max_len,
            temperature=temperature,
            num_generations=num_generations,
        )[0]
        return [ReprFactory.from_tensor(gen, repr_id=instance.repr_id).to_ink() for gen in gens]
