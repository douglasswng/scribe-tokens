from typing import Sequence

import torch
from torch import Tensor

from ml_model.locals.local import KVCaches, LocalModel
from ml_model.modules.decoder import TransformerDecoder
from ml_model.modules.embedder import CharEmbedder, Embedder, VectorEmbedder
from schemas.batch import Batch
from schemas.instance import IdMapper, Instance


class HTRModel(LocalModel):
    def __init__(self, repr_embedder: Embedder, decoder: TransformerDecoder | None = None):
        super().__init__()
        self._repr_embedder = repr_embedder
        self._char_embedder = CharEmbedder()
        self._decoder = decoder or TransformerDecoder()

        if isinstance(self._repr_embedder, VectorEmbedder):
            self._repr_embedder.strip_unembed()  # when using pretrained NTPModel

    def _losses(self, batch: Batch) -> dict[str, Tensor]:
        input, target, loss_mask = self._prepare_batch(
            batch=batch,
            prompt_embedder=self._repr_embedder,
            completion_embedder=self._char_embedder,
            prompt_attr="repr",
            completion_input_attr="char_input",
            completion_target_attr="char_target",
        )
        logits = self._forward(input)
        assert isinstance(logits, Tensor)
        return {"ce": self.ce_loss(logits, target, loss_mask)}

    def _forward(
        self,
        input: Tensor,
        start_pos: int | Tensor = 0,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        attn_mask: Tensor | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KVCaches]:
        result = self._decoder(
            input,
            start_pos=start_pos,
            kv_caches=kv_caches,
            attn_mask=attn_mask,
            use_cache=use_cache,
        )
        if not use_cache:
            assert isinstance(result, Tensor)
            pred = self._char_embedder.unembed(result)
            return pred
        else:
            assert isinstance(result, tuple)
            output, new_kv_caches = result
            pred = self._char_embedder.unembed(output)
            return pred, new_kv_caches

    def monitor(self, batch: Batch) -> None:
        instance = batch.get_random_instance()
        text_true = instance.parsed.text
        text_pred = self.batch_predict_text([instance])[0]
        self._track_ink(
            ink=instance.parsed.ink,
            task="HWR",
            caption=f"True: {text_true} | Pred: {text_pred}",
        )

    @torch.no_grad()
    def batch_predict_text(
        self,
        instances: Sequence[Instance],
        max_len: int = 50,
    ) -> list[str]:
        """
        Predict text for multiple instances in a single batched call.

        Args:
            instances: Sequence of instances to predict text for
            max_len: Maximum generation length

        Returns:
            List of predicted text strings
        """
        if self.training:
            raise ValueError("batch_predict_text should only be called in eval mode")

        context = [self._repr_embedder.embed(instance.repr) for instance in instances]
        bos = instances[0].char_bos
        eos = instances[0].char_eos

        batch_gens = self._generate_sequences(
            context=context,
            bos=bos,
            eos=eos,
            output_embedder=self._char_embedder,
            max_len=max_len,
            temperature=0.0,  # greedy sampling
            num_generations=1,  # single generation
        )

        return [IdMapper.ids_to_str(gens[0].tolist()) for gens in batch_gens]
