import torch
from torch import Tensor

from dataloader.dataset import IdMapper
from ml_model.locals.local import KVCaches, LocalModel
from ml_model.modules.decoder import TransformerDecoder
from ml_model.modules.embedder import CharEmbedder, Embedder
from schemas.batch import Batch
from schemas.instance import Instance


class HTRModel(LocalModel):
    def __init__(self, repr_embedder: Embedder, decoder: TransformerDecoder | None = None):
        super().__init__()
        self._repr_embedder = repr_embedder
        self._char_embedder = CharEmbedder()
        self._decoder = decoder or TransformerDecoder()

    def _losses(self, batch: Batch) -> dict[str, Tensor]:
        input, target, mask = self._prepare_batch(
            batch=batch,
            context_embedder=self._repr_embedder,
            target_embedder=self._char_embedder,
            context_attr="repr",
            target_input_attr="char_input",
            target_target_attr="char_target",
        )
        logits = self._forward(input)
        assert isinstance(logits, Tensor)
        return {"ce": self.ce_loss(logits, target, mask)}

    def _forward(
        self,
        input: Tensor,
        start_pos: int = 0,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KVCaches]:
        result = self._decoder(input, start_pos=start_pos, kv_caches=kv_caches, use_cache=use_cache)
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
        text_pred = self.predict_text(instance)
        self._track_ink(
            ink=instance.parsed.ink,
            task="HWR",
            caption=f"True: {text_true} | Pred: {text_pred}",
        )

    @torch.inference_mode()
    def predict_text(self, instance: Instance, max_len: int = 50) -> str:
        context = self._repr_embedder.embed(instance.repr)
        char_ids = self._generate_sequences(
            context=context,
            output_embedder=self._char_embedder,
            bos=instance.char_bos,
            eos=instance.char_eos,
            max_len=max_len,
            temperature=0.0,
            num_generations=1,
        )[0]
        return IdMapper.ids_to_str(char_ids.tolist())


if __name__ == "__main__":
    from dataloader.create import create_dataloaders
    from ml_model.factory import create_embedder
    from ml_model.id import ModelId, Task
    from utils.distributed_context import distributed_context

    for model_id in ModelId.create_task_model_ids(Task.HTR)[:]:
        print(model_id)
        train_loader, val_loader, test_loader = create_dataloaders(
            model_id=model_id,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        repr_embedder = create_embedder(model_id.repr_id)
        model = HTRModel(repr_embedder=repr_embedder).to(distributed_context.device)
        for batch in train_loader:
            model.train()
            losses = model(batch)
            print(losses)
            model.eval()
            model.monitor(batch)
            break
