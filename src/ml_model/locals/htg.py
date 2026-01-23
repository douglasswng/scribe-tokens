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
    def __init__(self, repr_embedder: Embedder):
        super().__init__()
        self._repr_embedder = repr_embedder
        self._char_embedder = CharEmbedder()
        self._decoder = TransformerDecoder()

    def _losses(self, batch: Batch) -> dict[str, Tensor]:
        input, target, mask = self._prepare_batch(
            batch=batch,
            context_embedder=self._char_embedder,
            target_embedder=self._repr_embedder,
            context_attr="char",
            target_input_attr="repr_input",
            target_target_attr="repr_target",
        )

        pred = self._forward(input)
        match pred:
            case Tensor():
                return {"ce": self.ce_loss(pred, target, mask)}
            case tuple() if len(pred) == 5:
                return {"nll": self.nll_loss(pred, target, mask)}
            case _:
                raise ValueError(f"Unsupported prediction type: {type(pred)}")

    def _forward(
        self,
        input: Tensor,
        start_pos: int = 0,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
    ) -> ForwardOutput | tuple[ForwardOutput, KVCaches]:
        result = self._decoder(input, start_pos=start_pos, kv_caches=kv_caches, use_cache=use_cache)
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
        inks_pred = self.generate_inks(instance, num_generations=1)
        self._track_ink(ink=inks_pred[0], task="HWG", caption=f"Text: {instance.parsed.text}")

    @torch.inference_mode()
    def generate_inks(
        self, instance: Instance, num_generations: int = 1, max_len: int = 500
    ) -> list[DigitalInk]:
        context = self._char_embedder.embed(instance.char)
        gens = self._generate_sequences(
            context=context,
            output_embedder=self._repr_embedder,
            bos=instance.repr_bos,
            eos=instance.repr_eos,
            max_len=max_len,
            temperature=1.0,
            num_generations=num_generations,
        )
        return [ReprFactory.from_tensor(gen, repr_id=instance.repr_id).to_ink() for gen in gens]


if __name__ == "__main__":
    from dataloader.create import create_dataloaders
    from ml_model.factory import create_embedder
    from ml_model.id import ModelId, Task
    from utils.distributed_context import distributed_context

    for model_id in ModelId.create_task_model_ids(Task.HTG)[::]:
        print(model_id)
        train_loader, val_loader, test_loader = create_dataloaders(
            model_id=model_id,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        repr_embedder = create_embedder(model_id.repr_id)
        model = HTGModel(repr_embedder=repr_embedder).to(distributed_context.device)
        for batch in train_loader:
            model.train()
            losses = model(batch)
            print(losses)
            model.eval()
            model.monitor(batch)
            break
