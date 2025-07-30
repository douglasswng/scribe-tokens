from functools import partial

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from core.model import LocalModel, ModelId
from core.data_schema import Batch, DigitalInk, PairBatch, InstancePair
from repr.factory import DefaultReprFactory
from model.modules.embedder import Embedder, CharEmbedder, TokenEmbedder
from model.modules.decoder import TransformerDecoder


class GenerationModel(LocalModel):
    def __init__(self, model_id: ModelId):
        super().__init__()
        self._model_id = model_id

        self._repr_embedder: Embedder = TokenEmbedder()
        self._char_embedder: Embedder = CharEmbedder()
        self._decoder = TransformerDecoder()

    def _forward(self, input: torch.Tensor) -> Tensor:
        output = self._decoder(input)
        logits = self._repr_embedder.unembed(output)
        return logits
    
    def _ce_loss(self,
                 pred: Tensor,
                 target: Tensor,
                 padding: Tensor) -> Tensor:
        logits_flat = pred.reshape(-1, pred.size(-1))
        target_flat = target.reshape(-1)
        padding_flat = padding.reshape(-1)
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(logits_flat, target_flat)
        masked_loss = loss * padding_flat
        return masked_loss.sum() / padding_flat.sum()
    
    def _prepare_context(self, instance_pair: InstancePair) -> Tensor:
        ref_repr = instance_pair.ref_instance.repr
        ref_repr_embed = self._repr_embedder.embed(ref_repr)

        main_char_input = instance_pair.main_instance.char_input
        main_char_embed = self._char_embedder.embed(main_char_input)

        context_embed = torch.cat([ref_repr_embed, main_char_embed])
        return context_embed
    
    def _prepare_batch(self, batch: PairBatch) -> tuple[Tensor, Tensor, Tensor]:
        inputs = []
        targets = []
        paddings = []
        for instance_pair in batch.datapoints:
            context_embed = self._prepare_context(instance_pair)

            main_repr_input = instance_pair.main_instance.repr_input
            main_repr_input_embed = self._repr_embedder(main_repr_input)

            main_repr_target = instance_pair.main_instance.repr_target
            main_repr_target_embed = self._repr_embedder(main_repr_target)

            input = torch.cat([context_embed, main_repr_input_embed])
            target = torch.cat([context_embed, main_repr_target_embed])

            context_len = context_embed.size(0)
            target_len = main_repr_target.size(0)
            padding = torch.cat([torch.zeros(context_len), torch.ones(target_len)])

            inputs.append(input)
            targets.append(target)
            paddings.append(padding)
        input = pad_sequence(inputs, batch_first=True, padding_value=0)
        target = pad_sequence(targets, batch_first=True, padding_value=0)
        padding = pad_sequence(paddings, batch_first=True, padding_value=0)
        return input, target, padding

    def losses(self, batch: Batch) -> dict[str, Tensor]:
        assert isinstance(batch, PairBatch)
        input, target, padding = self._prepare_batch(batch)
        main_repr_pred = self._forward(input)
        loss = self._ce_loss(main_repr_pred, target, padding)
        return {'ntp_ce': loss}
    
    def _generate_next_token(self, input: Tensor, temperature: float = 1.0) -> Tensor:
        gen_repr_pred = self._forward(input=input)
        last_pred = gen_repr_pred[:, -1]
        token_probs = torch.softmax(last_pred / temperature, dim=-1)
        token_indices = torch.multinomial(token_probs, 1)  # Keep shape [batch_size, 1]
        return token_indices
    
    def _is_end(self, gen_repr: Tensor, max_len: int=10) -> bool:  # TODO: more advanced
        seq_len = gen_repr.size(1)
        return seq_len >= max_len
    
    def _generate_tensor(self, instance_pair: InstancePair,
                         num_gen: int = 1,
                         temperature: float = 1.0) -> Tensor:
        context = self._prepare_context(instance_pair).unsqueeze(0).expand(num_gen, -1, -1)
        gen_repr = context[:, :1]  # bos token
        while not self._is_end(gen_repr):
            input = torch.cat([context, gen_repr], dim=1)
            next_token = self._generate_next_token(input=input,
                                                   temperature=temperature)
            next
            gen_repr = torch.cat([gen_repr, next_token], dim=1)
        return gen_repr
        
    def generate_inks(self, instance_pair: InstancePair,
                      num_gen: int = 1,
                      temperature: float = 1.0) -> list[DigitalInk]:
        if self.training:
            raise ValueError("Generation is not supported in training mode")

        ink_callable = partial(DefaultReprFactory.tensor_to_ink, id=self._model_id.repr_id)
        with torch.no_grad():
            tensor = self._generate_tensor(instance_pair=instance_pair,
                                           num_gen=num_gen,
                                           temperature=temperature)
            inks = [ink_callable(tensor=gen_repr) for gen_repr in tensor]
            return inks
    
    def monitor(self, batch: Batch) -> None:
        assert isinstance(batch, PairBatch)
        sample = batch.get_random_sample()
        instance_pair = sample.datapoints[0]
        ink = self.generate_inks(instance_pair=instance_pair)[0]
        ink.visualise(name=instance_pair.main_instance.parsed.text)

        
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
        model.eval()
        for batch in train_loader:
            model.monitor(batch)
            break