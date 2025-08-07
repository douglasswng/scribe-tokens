import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from core.data_schema import Batch
from model.modules.embedder import Embedder, CharEmbedder


class BatchPreper:
    def __init__(self, repr_embedder: Embedder, char_embedder: CharEmbedder):
        self._repr_embedder = repr_embedder
        self._char_embedder = char_embedder

    @property
    def _device(self) -> torch.device:
        return next(self._repr_embedder.parameters()).device

    def prepare_recog_batch(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
        repr_inputs = [self._repr_embedder.embed(instance.repr) for instance in batch.instances]
        char_inputs = [self._char_embedder.embed(instance.char_input) for instance in batch.instances]
        inputs = [torch.cat([repr, char], dim=0) for repr, char in zip(repr_inputs, char_inputs)]

        repr_targets = [instance.repr for instance in batch.instances]
        char_targets = [instance.char_target for instance in batch.instances]
        targets = [torch.cat([repr, char], dim=0) for repr, char in zip(repr_targets, char_targets)]

        repr_input_masks = [torch.zeros(repr.shape[0]) for repr in repr_inputs]
        char_target_masks = [torch.ones(char.shape[0]) for char in char_targets]
        target_masks = [torch.cat([repr_mask, char_target_mask], dim=0)
                        for repr_mask, char_target_mask in zip(repr_input_masks, char_target_masks)]

        return (self._pad_tensors(inputs),
                self._pad_tensors(targets),
                self._pad_tensors(target_masks).to(self._device))

    def prepare_gen_batch(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
        ...

    def _pad_tensors(self, tensors: list[Tensor]) -> Tensor:
        return pad_sequence(tensors, batch_first=True, padding_value=0)