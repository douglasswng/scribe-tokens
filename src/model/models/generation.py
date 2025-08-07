from functools import partial

import torch
from torch import Tensor
import torch.nn as nn

from core.model import LocalModel, ModelId
from core.data_schema import Batch, DigitalInk, Instance
from repr.factory import DefaultReprFactory
from model.modules.embedder import Embedder, CharEmbedder
from model.modules.decoder import TransformerDecoder
from model.models.batch_utils import BatchPreper
from model.models.loss_mixin import LossMixin


class GenerationModel(LocalModel, LossMixin):
    def __init__(self, model_id: ModelId, repr_embedder: Embedder):
        super().__init__()
        self._repr_embedder = repr_embedder
        self._char_embedder = CharEmbedder()

        self._decoder = TransformerDecoder()

        self._ink_callable = partial(DefaultReprFactory.tensor_to_ink, id=model_id.repr_id)
        self._batch_preper = BatchPreper(repr_embedder=repr_embedder, char_embedder=self._char_embedder)

    def _extend_char_mask(self, input: Tensor, char_mask: Tensor) -> Tensor:
        _, input_seq_len = input.shape
        _, char_seq_len = char_mask.shape
        
        extended_char_mask = torch.zeros_like(input, dtype=torch.bool)
        
        min_seq_len = min(input_seq_len, char_seq_len)
        extended_char_mask[:, :min_seq_len] = char_mask[:, :min_seq_len]
        
        return extended_char_mask

    def _forward(self, input: torch.Tensor, target_mask: Tensor) -> Tensor:
        pred = self._decoder(input)
        pred = pred * target_mask.unsqueeze(-1)
        logits = self._char_embedder.unembed(pred)
        return logits

    def losses(self, batch: Batch) -> dict[str, Tensor]:
        input, target, target_mask = self._batch_preper.prepare_gen_batch(batch)
        logits = self._forward(input, target_mask)
        loss = self.ce_loss(logits, target, target_mask)
        return {'ce': loss}

    def _generate_next_tensor(self, static_input: Tensor,
                              gen_tensors: Tensor,
                              temperature: float = 1.0) -> Tensor:
        ...

    def _terminate_generation(self, gen_tensors: Tensor, eos: Tensor) -> bool:
        ...
        
    def generate_inks(self, instance: Instance,
                      num_gen: int = 1,
                      temperature: float = 1.0,
                      max_len: int = 1000) -> list[DigitalInk]:
        if self.training:
            raise ValueError("Generation is not supported in training mode")
        
        with torch.no_grad():
            static_input = self._char_embedder.embed(instance.char)

            gen_tensors = instance.repr_bos.unsqueeze(0).expand(num_gen, -1)
            for _ in range(max_len):
                next_tensor = self._generate_next_tensor(
                    static_input=static_input,
                    gen_tensors=gen_tensors,
                    temperature=temperature
                )
                
                gen_tensors = torch.cat([gen_tensors, next_tensor], dim=1)

                if self._terminate_generation(gen_tensors, instance.repr_eos):
                    break

            inks = [self._ink_callable(tensor=gen_tesnor) for gen_tesnor in gen_tensors]
            return inks
    
    def monitor(self, batch: Batch) -> None:
        instance = batch.get_random_instance()
        ink = self.generate_inks(instance=instance)[0]
        ink.visualise(name=instance.parsed.text)

        
if __name__ == "__main__":
    from core.model import Task
    from dataloader.create import create_dataloaders
    from core.utils import distributed_context

    for model_id in ModelId.create_task_defaults(Task.GENERATION):
        train_loader, val_loader, test_loader = create_dataloaders(
            model_id=model_id,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        model = GenerationModel(model_id).to(distributed_context.device)
        for batch in train_loader:
            model.train()
            losses = model.losses(batch)
            print(f"Losses: {losses}")

            model.eval()
            model.monitor(batch)
            break