from typing import Sequence

import torch
from torch import Tensor

from ink_repr.factory import ReprFactory
from ml_model.locals.local import ForwardOutput, KVCaches, LocalModel
from ml_model.modules.decoder import TransformerDecoder
from ml_model.modules.embedder import CharEmbedder, Embedder
from schemas.batch import Batch
from schemas.ink import DigitalInk
from schemas.instance import Instance


class HTGModel(LocalModel):
    def __init__(self, repr_embedder: Embedder, decoder: TransformerDecoder | None = None):
        super().__init__()
        self._repr_embedder = repr_embedder
        self._char_embedder = CharEmbedder()
        self._decoder = decoder or TransformerDecoder()

    def _losses(self, batch: Batch) -> dict[str, Tensor]:
        input, target, loss_mask = self._prepare_batch(
            batch=batch,
            prompt_embedder=self._char_embedder,
            completion_embedder=self._repr_embedder,
            prompt_attr="char",
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
        result = self._decoder(
            input,
            start_pos=start_pos,
            kv_caches=kv_caches,
            attn_mask=attn_mask,
            use_cache=use_cache,
        )
        if not use_cache:
            assert isinstance(result, Tensor)
            pred = self._repr_embedder.unembed(result)
            return pred
        else:
            assert isinstance(result, tuple)
            output, new_kv_caches = result
            pred = self._repr_embedder.unembed(output)
            return pred, new_kv_caches

    def monitor(self, batch: Batch) -> None:
        instance = batch.get_random_instance()
        gen_inks = self.batch_generate_inks([instance], num_generations=1)
        self._track_ink(ink=gen_inks[0][0], task="HWG", caption=f"Text: {instance.parsed.text}")

    @torch.no_grad()
    def batch_generate_inks(
        self,
        instances: Sequence[Instance],
        max_len: int = 500,
        temperature: float = 1.0,
        num_generations: int = 1,
    ) -> list[list[DigitalInk]]:
        """
        Generate inks for multiple instances in a single batched call.

        Args:
            instances: Sequence of instances to generate inks for
            num_generations: Number of generations per instance
            max_len: Maximum generation length
            temperature: Sampling temperature

        Returns:
            List of lists: outer list per instance, inner list of num_generations DigitalInk
        """
        if self.training:
            raise ValueError("batch_predict_text should only be called in eval mode")

        context = [self._char_embedder.embed(instance.char) for instance in instances]
        bos = instances[0].repr_bos
        eos = instances[0].repr_eos

        batch_gens = self._generate_sequences(
            context=context,
            bos=bos,
            eos=eos,
            output_embedder=self._repr_embedder,
            max_len=max_len,
            temperature=temperature,
            num_generations=num_generations,
        )

        return [
            [ReprFactory.from_tensor(gen, repr_id=instance.repr_id).to_ink() for gen in gens]
            for instance, gens in zip(instances, batch_gens)
        ]
