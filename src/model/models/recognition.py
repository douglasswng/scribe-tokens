import torch
from torch import Tensor

from core.data_schema import Batch, DigitalInk, IdMapper, Instance, SingletonBatch
from core.model import LocalModel, ModelId, Tracker
from model.models.batch_utils import BatchPreper
from model.models.loss_mixin import LossMixin
from model.modules.decoder import TransformerDecoder
from model.modules.embedder import CharEmbedder, Embedder


class RecognitionModel(LocalModel, LossMixin):
    def __init__(
        self, model_id: ModelId, repr_embedder: Embedder, decoder: TransformerDecoder | None = None
    ):
        super().__init__()
        self._model_id = model_id
        self._repr_embedder = repr_embedder
        self._char_embedder = CharEmbedder()

        self._decoder = decoder or TransformerDecoder()

        self._batch_preper = BatchPreper(
            task=model_id.task, repr_embedder=repr_embedder, char_embedder=self._char_embedder
        )

    def _forward(self, input: Tensor) -> Tensor:
        pred = self._decoder(input)
        logits = self._char_embedder.unembed(pred)
        return logits

    def losses(self, batch: Batch) -> dict[str, Tensor]:
        input, target, target_mask = self._batch_preper.prepare_batch(batch)
        logits = self._forward(input)
        return {"ce": self.ce_loss(logits, target, target_mask)}

    def _generate_next_id(self, static_input: Tensor, generated_ids: list[int]) -> int:
        if generated_ids:
            gen_tensor = torch.tensor(generated_ids, device=static_input.device)
            gen_embedded = self._char_embedder.embed(gen_tensor)
            current_input = torch.cat([static_input, gen_embedded], dim=0)
        else:
            current_input = static_input

        logits = self._forward(current_input.unsqueeze(0))
        next_id = torch.argmax(logits[0, -1], dim=-1)
        return int(next_id)

    def predict_text(self, instance: Instance, max_len: int = 100) -> str:
        if self.training:
            raise ValueError("Prediction is not supported in training mode")

        with torch.no_grad():
            repr_embedded = self._repr_embedder.embed(instance.repr)
            bos_embedded = self._char_embedder.embed(instance.char_bos.unsqueeze(0))
            static_input = torch.cat([repr_embedded, bos_embedded], dim=0)

            generated_ids = []
            for _ in range(max_len):
                next_id = self._generate_next_id(
                    static_input=static_input, generated_ids=generated_ids
                )

                if next_id == instance.char_eos.item():
                    break

                generated_ids.append(next_id)

            return IdMapper.ids_to_str(generated_ids)

    def monitor(self, batch: Batch, tracker: Tracker | None = None) -> None:
        assert isinstance(batch, SingletonBatch)
        instance = batch.get_random_instance()
        text_pred = self.predict_text(instance)
        text_true = instance.parsed.text
        self._monitor_ink(
            instance.parsed.ink, "OCR", f"True: {text_true} | Pred: {text_pred}", tracker
        )

    def _monitor_ink(
        self, ink: DigitalInk, task: str, caption: str, tracker: Tracker | None = None
    ) -> None:
        ink.visualise(name=f"{task}: {caption}")
        if tracker is not None:
            tracker.log_image(ink.to_image(), task, caption)


if __name__ == "__main__":
    from core.utils import distributed_context

    from core.model import ModelId, Task
    from dataloader.create import create_dataloaders
    from model.factory import ReprEmbedderFactory

    for model_id in ModelId.create_task_model_ids(Task.RECOGNITION)[::-1]:
        print(model_id)
        train_loader, val_loader, test_loader = create_dataloaders(
            model_id=model_id,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        repr_embedder = ReprEmbedderFactory.create(model_id.repr_id)
        model = RecognitionModel(model_id=model_id, repr_embedder=repr_embedder).to(
            distributed_context.device
        )
        for batch in train_loader:
            model.train()
            losses = model.losses(batch)
            print(losses)
            model.eval()
            model.monitor(batch)
            break
