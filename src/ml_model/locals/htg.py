import torch
from torch import Tensor

from ink_repr.factory import ReprFactory
from ml_model.locals.local import LocalModel
from ml_model.modules.decoder import TransformerDecoder
from ml_model.modules.embedder import CharEmbedder, Embedder, MDNOutput
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
        if batch.instances[0].is_token:
            assert isinstance(pred, Tensor)
            return {"ce": self.ce_loss(pred, target, mask)}
        else:
            assert isinstance(pred, tuple)
            return {"nll": self.nll_loss(pred, target, mask)}

    def _forward(self, input: Tensor) -> Tensor | MDNOutput:
        output = self._decoder(input)
        return self._repr_embedder.unembed(output)

    def monitor(self, batch: Batch) -> None:
        instance = batch.get_random_instance()
        ink_pred = self.generate_ink(instance)
        self._track_ink(ink=ink_pred, task="HWG", caption=f"Text: {instance.parsed.text}")

    @torch.inference_mode()  # TODO: batch generation
    def generate_ink(self, instance: Instance, max_len: int = 50) -> DigitalInk:
        context = self._char_embedder.embed(instance.char)
        gen = self._generate_sequence(
            context=context,
            output_embedder=self._repr_embedder,
            bos=instance.repr_bos,
            eos=instance.repr_eos,
            max_len=max_len,
            temperature=1.0,
        )
        return ReprFactory.from_tensor(gen, repr_id=instance.repr_id).to_ink()


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
