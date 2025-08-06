import torch
from torch import Tensor

from core.model import LocalModel
from core.data_schema import Batch, Instance, IdMapper
from model.modules.embedder import CharEmbedder, Embedder
from model.modules.decoder import TransformerDecoder


class RecognitionModel(LocalModel):
    def __init__(self, repr_embedder: Embedder):
        super().__init__()
        self._repr_embedder = repr_embedder
        self._char_embedder = CharEmbedder()

        self._decoder = TransformerDecoder()

    def _prepare_batch(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
        reprs = [self._repr_embedder.embed(instance.repr) for instance in batch.instances]
        char_inputs = [self._char_embedder.embed(instance.char_input) for instance in batch.instances]
        inputs = [torch.cat([repr, char], dim=0) for repr, char in zip(reprs, char_inputs)]

        char_targets = [self._char_embedder.embed(instance.char_target) for instance in batch.instances]
        targets = [torch.cat([repr, char], dim=0) for repr, char in zip(reprs, char_targets)]

        repr_masks = [torch.zeros(repr.shape[0]) for repr in reprs]
        char_target_masks = [torch.ones(char.shape[0]) for char in char_targets]
        target_masks = [torch.cat([repr_mask, char_target_mask], dim=0)
                        for repr_mask, char_target_mask in zip(repr_masks, char_target_masks)]

        return (self._decoder.pad_tensors(inputs),
                self._decoder.pad_tensors(targets),
                self._decoder.pad_tensors(target_masks).to(self.device))

    def losses(self, batch: Batch) -> dict[str, Tensor]:
        input, target, target_mask = self._prepare_batch(batch)
        pred = self._decoder(input)
        return {'ce': self._decoder.ce_loss(pred, target, target_mask)}

    def _generate_next_char(self, input: Tensor) -> Tensor:
        last_logits = self._decoder.next_logits(input)[0]
        next_char = torch.argmax(last_logits, dim=-1)
        return next_char

    def predict_text(self, instance: Instance, max_len: int=100) -> str:
        repr = instance.repr
        char_bos = instance.char_bos
        char_eos = instance.char_eos

        base_input = torch.cat([self._repr_embedder.embed(repr), self._char_embedder.embed(char_bos)], dim=0)
        gen = torch.tensor([])
        while True:
            input = torch.cat([base_input, gen], dim=0)
            next_char = self._generate_next_char(input)
            if next_char.item() == char_eos.item() or gen.size(0) > max_len:
                break
            gen = torch.cat([gen, next_char], dim=0)
        return IdMapper.ids_to_str(gen.tolist())

    def monitor(self, batch: Batch) -> None:
        instance = batch.get_random_instance()
        text_pred = self.predict_text(instance)
        instance.parsed.ink.visualise(name=text_pred)