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

        return (self._decoder.pad_tensors(inputs),
                self._decoder.pad_tensors(targets),
                self._decoder.pad_tensors(target_masks).to(self.device))

    def _forward(self, input: Tensor, target_mask: Tensor) -> Tensor:
        pred = self._decoder(input)
        pred = pred * target_mask.unsqueeze(-1)
        logits = self._char_embedder.unembed(pred)
        return logits

    def losses(self, batch: Batch) -> dict[str, Tensor]:
        input, target, target_mask = self._prepare_batch(batch)
        logits = self._forward(input, target_mask)
        return {'ce': self._decoder.ce_loss(logits, target, target_mask)}

    def _generate_next_char(self, input: Tensor, target_mask: Tensor) -> Tensor:
        logits = self._forward(input, target_mask)
        last_logits = logits[0, -1]
        next_char = torch.argmax(last_logits, dim=-1)
        return next_char

    def predict_text(self, instance: Instance, max_len: int=100) -> str:
        with torch.no_grad():
            repr = instance.repr
            char_bos = instance.char_bos
            char_eos = instance.char_eos
            
            device = repr.device  # Get device from input tensors
            
            repr_input = self._repr_embedder.embed(repr)
            char_input = self._char_embedder.embed(char_bos.unsqueeze(0))  # Add sequence dimension
            
            # Initialize generation tensor with proper device and dtype
            gen = torch.empty(0, dtype=char_bos.dtype, device=device)
            
            while True:
                # Embed the generated characters so far
                if len(gen) > 0:
                    gen_embedded = self._char_embedder.embed(gen)
                    input = torch.cat([repr_input, char_input, gen_embedded], dim=0)
                else:
                    input = torch.cat([repr_input, char_input], dim=0)
                
                # Create proper target mask
                repr_mask = torch.zeros(repr_input.shape[0], device=device)
                char_mask = torch.zeros(char_input.shape[0], device=device)
                gen_mask = torch.ones(len(gen), device=device)
                target_mask = torch.cat([repr_mask, char_mask, gen_mask], dim=0)
                
                # Ensure input has batch dimension for the decoder
                input_batch = input.unsqueeze(0)  # Add batch dimension
                target_mask_batch = target_mask.unsqueeze(0)  # Add batch dimension
                
                next_char = self._generate_next_char(input_batch, target_mask_batch)
                next_char = next_char.squeeze(0)  # Remove batch dimension
                
                if next_char.item() == char_eos.item() or len(gen) >= max_len:
                    break
                    
                gen = torch.cat([gen, next_char.unsqueeze(0)], dim=0)
            
            return IdMapper.ids_to_str(gen.tolist())

    def monitor(self, batch: Batch) -> None:
        instance = batch.get_random_instance()
        text_pred = self.predict_text(instance)
        instance.parsed.ink.visualise(name=text_pred)