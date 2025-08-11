import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from core.data_schema import Batch, SingletonBatch, PairBatch
from core.model import Task
from model.modules.embedder import Embedder, CharEmbedder


class BatchPreper:
    def __init__(self, task: Task, repr_embedder: Embedder, char_embedder: CharEmbedder):
        self._task = task
        self._repr_embedder = repr_embedder
        self._char_embedder = char_embedder

    @property
    def _device(self) -> torch.device:
        return next(self._repr_embedder.parameters()).device

    def prepare_batch(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
        match self._task:
            case Task.RECOGNITION:
                assert isinstance(batch, SingletonBatch)
                return self._prepare_recog_batch(batch)
            case Task.GENERATION:
                assert isinstance(batch, PairBatch)
                return self._prepare_gen_batch(batch)
            case Task.PRETRAINING_NTP:
                assert isinstance(batch, SingletonBatch)
                return self._prepare_pretrain_batch(batch)
            case _:
                raise ValueError(f"Invalid task: {self._task}")

    def _prepare_pretrain_batch(self, batch: SingletonBatch) -> tuple[Tensor, Tensor, Tensor]:
        repr_embeddings = [self._repr_embedder.embed(inst.repr_input) for inst in batch.instances]
        inputs = self._pad_tensors(repr_embeddings)

        repr_targets = [inst.repr_target for inst in batch.instances]
        targets = self._pad_tensors(repr_targets)

        repr_masks = [self._create_bool_mask(tgt, value=True) for tgt in repr_targets]
        masks = self._pad_tensors(repr_masks)

        return inputs, targets, masks

    def _prepare_recog_batch(self, batch: SingletonBatch) -> tuple[Tensor, Tensor, Tensor]:
        repr_embeddings = [self._repr_embedder.embed(inst.repr) for inst in batch.instances]
        char_embeddings = [self._char_embedder.embed(inst.char_input) for inst in batch.instances]
        
        inputs = self._concat_and_pad(repr_embeddings, char_embeddings)
        
        repr_targets = [self._create_empty_target(inst.repr, inst.char_target)
                        for inst in batch.instances]
        char_targets = [inst.char_target for inst in batch.instances]
        targets = self._concat_and_pad(repr_targets, char_targets)
        
        repr_masks = [self._create_bool_mask(tgt, value=False) for tgt in repr_targets]
        char_masks = [self._create_bool_mask(tgt, value=True) for tgt in char_targets]
        masks = self._concat_and_pad(repr_masks, char_masks)
        
        return inputs, targets, masks

    def _prepare_gen_batch(self, batch: PairBatch) -> tuple[Tensor, Tensor, Tensor]:
        main_char_embeddings = [self._char_embedder.embed(inst.char) for inst in batch.main_instances]
        ref_char_embeddings = [self._char_embedder.embed(inst.char) for inst in batch.ref_instances]
        main_repr_embeddings = [self._repr_embedder.embed(inst.repr) for inst in batch.main_instances]
        ref_repr_embeddings = [self._repr_embedder.embed(inst.repr_input) for inst in batch.ref_instances]
        inputs = self._concat_and_pad(ref_repr_embeddings, ref_char_embeddings, main_char_embeddings, main_repr_embeddings)
        
        ref_repr_targets = [self._create_empty_target(inst.repr, inst.repr)
                            for inst in batch.ref_instances]
        ref_char_targets = [self._create_empty_target(inst.char, inst.repr)
                            for inst in batch.ref_instances]
        main_char_targets = [self._create_empty_target(inst.char, inst.repr)
                            for inst in batch.main_instances]
        main_repr_targets = [inst.repr_target for inst in batch.main_instances]    
        targets = self._concat_and_pad(ref_repr_targets, ref_char_targets, main_char_targets, main_repr_targets)
        
        ref_repr_masks = [self._create_bool_mask(tgt, value=False) for tgt in ref_repr_targets]
        ref_char_masks = [self._create_bool_mask(tgt, value=False) for tgt in ref_char_targets]
        main_char_masks = [self._create_bool_mask(tgt, value=False) for tgt in main_char_targets]
        main_repr_masks = [self._create_bool_mask(tgt, value=True) for tgt in main_repr_targets]
        masks = self._concat_and_pad(ref_repr_masks, ref_char_masks, main_char_masks, main_repr_masks)
        
        return inputs, targets, masks

    def _create_empty_target(self, orig_target: Tensor, ref_target: Tensor) -> Tensor:
        if ref_target.dtype in {torch.int, torch.long}:
            return self._create_bool_mask(orig_target, value=False).int()

        assert ref_target.dtype == torch.float

        ref_dim = ref_target.shape[-1]

        assert ref_dim == 5  # only Point-5 should require this logic

        orig_len = orig_target.shape[0]
        return torch.zeros(orig_len, ref_dim).to(self._device)

    def _create_bool_mask(self, tensor: Tensor, value: bool) -> Tensor:
        if value:
            return torch.ones(tensor.shape[0], dtype=torch.bool).to(self._device)
        else:
            return torch.zeros(tensor.shape[0], dtype=torch.bool).to(self._device)

    def _concat_and_pad(self, *vector_lists: list[Tensor]) -> Tensor:
        combined = [torch.cat(tensors, dim=0) 
                    for tensors in zip(*vector_lists)]
        return self._pad_tensors(combined)

    def _pad_tensors(self, tensors: list[Tensor]) -> Tensor:
        return pad_sequence(tensors, batch_first=True, padding_value=0)