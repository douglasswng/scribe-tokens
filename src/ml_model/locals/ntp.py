import torch
from torch import Tensor

from ink_repr.factory import ReprFactory
from ml_model.locals.local import LocalModel
from ml_model.modules.decoder import TransformerDecoder
from ml_model.modules.embedder import Embedder
from schemas.batch import Batch
from schemas.ink import DigitalInk
from schemas.instance import Instance


class NTPModel(LocalModel):
    """Next Token Prediction model that predicts next repr given previous repr."""

    def __init__(self, repr_embedder: Embedder):
        super().__init__()
        self._repr_embedder = repr_embedder
        self._decoder = TransformerDecoder()

    def _losses(self, batch: Batch) -> dict[str, Tensor]:
        input, target, mask = self._prepare_batch_tensors(
            batch=batch,
            context_embedder=None,
            target_embedder=self._repr_embedder,
            context_attr=None,
            target_input_attr="repr_input",
            target_target_attr="repr_target",
        )

        output = self._decoder(input)
        pred = self._repr_embedder.unembed(output)
        if batch.instances[0].is_token:
            assert isinstance(pred, Tensor)
            return {"ce": self.ce_loss(pred, target, mask)}
        else:
            assert isinstance(pred, tuple)
            return {"nll": self.nll_loss(pred, target, mask)}

    def monitor(self, batch: Batch) -> None:
        instance = batch.get_random_instance()
        ink_true = ReprFactory.from_tensor(instance.repr, repr_id=instance.repr_id).to_ink()
        ink_pred = self.generate_ink(instance)
        
        self._track_ink(
            ink=ink_true,
            task="NTP",
            caption=f"True: {instance.parsed.text}",
        )
        self._track_ink(
            ink=ink_pred,
            task="NTP",
            caption=f"Pred: {instance.parsed.text}",
        )

    @torch.inference_mode()
    def generate_ink(self, instance: Instance, max_len: int = 50) -> DigitalInk:
        if self.training:
            raise ValueError("Prediction is not supported in training mode")

        gen = instance.repr_bos.unsqueeze(0)  # [1, 5] (vector) or [1] (token)
        for _ in range(max_len):
            gen_embed = self._repr_embedder.embed(gen).unsqueeze(0)  # [1, gen_len, hidden_dim]
            output = self._decoder(gen_embed)  # [1, gen_len, hidden_dim]
            pred = self._repr_embedder.unembed(output)  # Tensor [1, gen_len, vocab] or MDNOutput
            
            if instance.is_token:
                assert isinstance(pred, Tensor)
                next_token = self.sample_token(pred[0, -1], temperature=1.0)  # [1]
                gen = torch.cat([gen, next_token], dim=0)  # [seq_len+1]
                if int(next_token) == int(instance.repr_eos):
                    break
            else:
                assert isinstance(pred, tuple)
                last_pred = tuple(tensor[0:1, -1] for tensor in pred)
                assert len(last_pred) == 5
                next_vector = self.sample_vector(last_pred, temperature=1.0)  # [1, 1, 5]
                next_vector = next_vector.squeeze(0).squeeze(0)  # [5]
                gen = torch.cat([gen, next_vector.unsqueeze(0)], dim=0)  # [seq_len+1, 5]
                if torch.any(next_vector * instance.repr_eos == 1.0):
                    break
        
        return ReprFactory.from_tensor(gen, repr_id=instance.repr_id).to_ink()


if __name__ == "__main__":
    from dataloader.create import create_dataloaders
    from ml_model.factory import create_embedder
    from ml_model.id import ModelId, Task
    from utils.distributed_context import distributed_context

    for model_id in ModelId.create_task_model_ids(Task.NTP)[::]:
        print(model_id)
        train_loader, val_loader, test_loader = create_dataloaders(
            model_id=model_id,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        repr_embedder = create_embedder(model_id.repr_id)
        model = NTPModel(repr_embedder=repr_embedder).to(distributed_context.device)
        for batch in train_loader:
            model.train()
            losses = model(batch)
            print(losses)
            model.eval()
            model.monitor(batch)
            break
