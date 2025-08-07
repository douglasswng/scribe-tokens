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
        ink_repr_embeddings = [self._repr_embedder.embed(instance.repr) for instance in batch.instances]
        char_input_embeddings = [self._char_embedder.embed(instance.char_input) for instance in batch.instances]
        recognition_inputs = [torch.cat([ink_emb, char_emb], dim=0) 
                             for ink_emb, char_emb in zip(ink_repr_embeddings, char_input_embeddings)]

        ink_repr_targets = [instance.repr for instance in batch.instances]
        char_prediction_targets = [instance.char_target for instance in batch.instances]
        recognition_targets = [torch.cat([ink_tgt, char_tgt], dim=0) 
                              for ink_tgt, char_tgt in zip(ink_repr_targets, char_prediction_targets)]

        ink_reconstruction_masks = [torch.zeros(ink_emb.shape[0]) for ink_emb in ink_repr_embeddings]
        char_prediction_masks = [torch.ones(char_tgt.shape[0]) for char_tgt in char_prediction_targets]
        recognition_loss_masks = [torch.cat([ink_mask, char_mask], dim=0)
                                 for ink_mask, char_mask in zip(ink_reconstruction_masks, char_prediction_masks)]

        return (self._pad_tensors(recognition_inputs),
                self._pad_tensors(recognition_targets),
                self._pad_tensors(recognition_loss_masks).to(self._device))

    def prepare_gen_batch(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
        char_embeddings = [self._char_embedder.embed(instance.char) for instance in batch.instances]
        ink_input_embeddings = [self._repr_embedder.embed(instance.repr_input) for instance in batch.instances]
        generation_inputs = [torch.cat([char_emb, ink_emb], dim=0) 
                            for char_emb, ink_emb in zip(char_embeddings, ink_input_embeddings)]

        char_reconstruction_targets = [instance.char for instance in batch.instances]
        ink_generation_targets = [instance.repr_target for instance in batch.instances]
        generation_targets = [torch.cat([char_tgt, ink_tgt], dim=0) 
                             for char_tgt, ink_tgt in zip(char_reconstruction_targets, ink_generation_targets)]

        char_reconstruction_masks = [torch.ones(char_emb.shape[0]) for char_emb in char_embeddings]
        ink_generation_masks = [torch.zeros(ink_tgt.shape[0]) for ink_tgt in ink_generation_targets]
        generation_loss_masks = [torch.cat([char_mask, ink_mask], dim=0)
                                for char_mask, ink_mask in zip(char_reconstruction_masks, ink_generation_masks)]

        return (self._pad_tensors(generation_inputs),
                self._pad_tensors(generation_targets),
                self._pad_tensors(generation_loss_masks).to(self._device))

    def _pad_tensors(self, tensors: list[Tensor]) -> Tensor:
        return pad_sequence(tensors, batch_first=True, padding_value=0)