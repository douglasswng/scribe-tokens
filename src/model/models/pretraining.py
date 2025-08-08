from core.model import ModelId
from model.modules.embedder import Embedder
from model.modules.decoder import TransformerDecoder
from model.models.generation import GenerationModel



class PretrainingModel(GenerationModel):
    def __init__(self, model_id: ModelId, repr_embedder: Embedder):
        super().__init__(model_id=model_id, repr_embedder=repr_embedder)

    @property
    def decoder(self) -> TransformerDecoder:
        return self._decoder

    @property
    def repr_embedder(self) -> Embedder:
        return self._repr_embedder

        
if __name__ == "__main__":
    from core.model import Task, ModelId
    from dataloader.create import create_dataloaders
    from model.factory import ReprEmbedderFactory
    from core.utils import distributed_context

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
        model = PretrainingModel(model_id=model_id, repr_embedder=repr_embedder).to(distributed_context.device)
        for batch in train_loader:
            model.train()
            losses = model.losses(batch)
            print(losses)
            model.eval()
            model.monitor(batch)
            break