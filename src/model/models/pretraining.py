from core.data_schema import Batch, SingletonBatch
from core.model import ModelId, Tracker
from model.models.generation import GenerationModel
from model.modules.decoder import TransformerDecoder
from model.modules.embedder import Embedder


class PretrainingModel(GenerationModel):
    def __init__(self, model_id: ModelId, repr_embedder: Embedder):
        super().__init__(model_id=model_id, repr_embedder=repr_embedder)

    @property
    def decoder(self) -> TransformerDecoder:
        return self._decoder

    @property
    def repr_embedder(self) -> Embedder:
        return self._repr_embedder

    def monitor(self, batch: Batch, tracker: Tracker | None = None) -> None:
        assert isinstance(batch, SingletonBatch)
        main_instance = batch.get_random_instance()
        gen_ink = self.generate_inks(main_instance=main_instance)[0]
        main_text = main_instance.parsed.text
        self._monitor_ink(gen_ink, "Generated", main_text, tracker)


if __name__ == "__main__":
    from core.utils import distributed_context

    from core.model import ModelId, Task
    from dataloader.create import create_dataloaders
    from model.factory import ReprEmbedderFactory

    for model_id in ModelId.create_task_model_ids(Task.PRETRAINING_NTP)[::-1]:
        print(model_id)
        train_loader, val_loader, test_loader = create_dataloaders(
            model_id=model_id,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        repr_embedder = ReprEmbedderFactory.create(model_id.repr_id)
        model = PretrainingModel(model_id=model_id, repr_embedder=repr_embedder).to(
            distributed_context.device
        )
        for batch in train_loader:
            model.train()
            losses = model.losses(batch)
            print(losses)
            model.eval()
            model.monitor(batch)
            break
