from functools import partial

import torch
from torch import Tensor
import torch.nn as nn

from core.model import LocalModel, ModelId
from core.data_schema import Batch, DigitalInk, Instance
from repr.factory import DefaultReprFactory
from model.modules.embedder import Embedder, CharEmbedder
from model.modules.decoder import TransformerDecoder


class GenerationModel(LocalModel):
    def __init__(self, model_id: ModelId, repr_embedder: Embedder):
        super().__init__()
        self._ink_callable = partial(DefaultReprFactory.tensor_to_ink, id=model_id.repr_id)

        self._repr_embedder = repr_embedder
        self._char_embedder = CharEmbedder()

        self._decoder = TransformerDecoder()

    def _extend_char_mask(self, input: Tensor, char_mask: Tensor) -> Tensor:
        _, input_seq_len = input.shape
        _, char_seq_len = char_mask.shape
        
        extended_char_mask = torch.zeros_like(input, dtype=torch.bool)
        
        min_seq_len = min(input_seq_len, char_seq_len)
        extended_char_mask[:, :min_seq_len] = char_mask[:, :min_seq_len]
        
        return extended_char_mask

    def _forward(self, input: torch.Tensor,
                 char_mask: Tensor) -> Tensor:
        char_mask = self._extend_char_mask(input=input, char_mask=char_mask)
        repr_mask = ~char_mask
        
        repr_input = input * repr_mask
        char_input = input * char_mask
        
        repr_embedded = self._repr_embedder.embed(repr_input)
        char_embedded = self._char_embedder.embed(char_input)

        repr_embedded = repr_embedded * repr_mask.unsqueeze(-1)
        char_embedded = char_embedded * char_mask.unsqueeze(-1)

        combined_input = repr_embedded + char_embedded
        
        output = self._decoder(combined_input)
        logits = self._repr_embedder.unembed(output)
        return logits
    
    def _ce_loss(self,
                 pred: Tensor,
                 target: Tensor,
                 mask: Tensor) -> Tensor:
        logits_flat = pred.reshape(-1, pred.size(-1))
        target_flat = target.reshape(-1)
        mask_flat = mask.reshape(-1)
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(logits_flat, target_flat)
        masked_loss = loss * mask_flat
        return masked_loss.sum() / mask_flat.sum()

    def losses(self, batch: Batch) -> dict[str, Tensor]:
        assert isinstance(batch, PairBatch)
        main_repr_pred = self._forward(
            input=batch.input,
            char_mask=batch.char_mask
        )
        loss = self._ce_loss(main_repr_pred, batch.target, batch.target_mask)
        return {'ntp_ce': loss}
    
    def _generate_next_token(self, input: Tensor, 
                             char_mask: Tensor,
                             temperature: float = 1.0) -> Tensor:
        gen_repr_pred = self._forward(
            input=input,
            char_mask=char_mask
        )
        last_pred = gen_repr_pred[:, -1]
        token_probs = torch.softmax(last_pred / temperature, dim=-1)
        token_indices = torch.multinomial(token_probs, 1)  # Keep shape [batch_size, 1]
        return token_indices
    
    def _is_end(self, gen_repr: Tensor, max_len: int=1000) -> bool:  # TODO: more advanced
        if gen_repr.size(1) > max_len:
            return True
        return False
    
    def _generate_tensor(self, instance_pair: InstancePair,
                         num_gen: int = 1,
                         temperature: float = 1.0) -> Tensor:
        context = instance_pair.context.unsqueeze(0).expand(num_gen, -1)
        char_mask = instance_pair.char_mask.unsqueeze(0).expand(num_gen, -1)
        gen_repr = context[:, :1]  # bos token
        
        while not self._is_end(gen_repr):
            input = torch.cat([context, gen_repr], dim=1)
            next_token = self._generate_next_token(
                input=input,
                char_mask=char_mask,
                temperature=temperature
            )
            gen_repr = torch.cat([gen_repr, next_token], dim=1)
        return gen_repr
        
    def generate_inks(self, instance: Instance,
                      num_gen: int = 1,
                      temperature: float = 1.0) -> list[DigitalInk]:
        if self.training:
            raise ValueError("Generation is not supported in training mode")
        
        with torch.no_grad():
            tensor = self._generate_tensor(instance_pair=instance_pair,
                                           num_gen=num_gen,
                                           temperature=temperature)
            inks = [self._ink_callable(tensor=gen_repr) for gen_repr in tensor]
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