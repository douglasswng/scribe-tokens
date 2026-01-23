import torch
from torch import Tensor

from dataloader.dataset import IdMapper
from ml_model.locals.local import LocalModel
from ml_model.modules.decoder import TransformerDecoder
from ml_model.modules.embedder import CharEmbedder, Embedder
from schemas.batch import Batch
from schemas.instance import Instance


class HWRModel(LocalModel):
    def __init__(self, repr_embedder: Embedder):
        super().__init__()
        self._repr_embedder = repr_embedder
        self._char_embedder = CharEmbedder()
        self._decoder = TransformerDecoder()

    def _losses(self, batch: Batch) -> dict[str, Tensor]:
        input, target, mask = self._prepare_batch_tensors(
            batch=batch,
            context_embedder=self._repr_embedder,
            target_embedder=self._char_embedder,
            context_attr="repr",
            target_input_attr="char_input",
            target_target_attr="char_target",
        )

        output = self._decoder(input)
        logits = self._char_embedder.unembed(output)

        return {"ce": self.ce_loss(logits, target, mask)}

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
    def predict_text(self, instance: Instance, max_len: int = 100) -> str:
        if self.training:
            raise ValueError("Prediction is not supported in training mode")

        input = self._repr_embedder.embed(instance.repr).unsqueeze(0)
        char_ids: list[int] = [int(instance.char_bos)]
        for _ in range(max_len):
            last_char = torch.tensor(char_ids[-1:], device=self._device, dtype=torch.long)
            last_char = self._char_embedder.embed(last_char).unsqueeze(0)  # [1, 1, hidden_dim]
            input = torch.cat([input, last_char], dim=1)  # [1, seq_len, hidden_dim]
            output = self._decoder(input)
            logits = self._char_embedder.unembed(output)  # [1, seq_len, num_chars]
            next_char = int(self.sample_token(logits[0, -1], temperature=0.0))
            char_ids.append(next_char)
            if next_char == int(instance.char_eos):
                break
        return IdMapper.ids_to_str(char_ids)


if __name__ == "__main__":
    from dataloader.create import create_dataloaders
    from ml_model.factory import create_embedder
    from ml_model.id import ModelId, Task
    from utils.distributed_context import distributed_context

    for model_id in ModelId.create_task_model_ids(Task.HWR)[::-1]:
        print(model_id)
        train_loader, val_loader, test_loader = create_dataloaders(
            model_id=model_id,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        repr_embedder = create_embedder(model_id.repr_id)
        model = HWRModel(repr_embedder=repr_embedder).to(distributed_context.device)
        for batch in train_loader:
            model.train()
            losses = model(batch)
            print(losses)
            model.eval()
            model.monitor(batch)
            break
