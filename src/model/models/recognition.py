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
    
    def _generate_next_id(self, static_input: Tensor, generated_ids: list[int]) -> Tensor:
        if generated_ids:
            gen_tensor = torch.tensor(generated_ids, device=static_input.device)
            gen_embedded = self._char_embedder.embed(gen_tensor)
            current_input = torch.cat([static_input, gen_embedded], dim=0)
        else:
            current_input = static_input

        target_mask = torch.cat([
            torch.zeros(static_input.shape[0], device=static_input.device),
            torch.ones(len(generated_ids), device=static_input.device)
        ], dim=0)

        logits = self._forward(current_input.unsqueeze(0), target_mask.unsqueeze(0))
        next_char = torch.argmax(logits[0, -1], dim=-1)
        return next_char

    def predict_text(self, instance: Instance, max_len: int = 100) -> str:
        if self.training:
            raise ValueError("Prediction is not supported in training mode")
            
        with torch.no_grad():
            repr_embedded = self._repr_embedder.embed(instance.repr)
            bos_embedded = self._char_embedder.embed(instance.char_bos.unsqueeze(0))
            static_input = torch.cat([repr_embedded, bos_embedded], dim=0)

            generated_ids = []
            for _ in range(max_len):
                next_id = self._generate_next_id(
                    static_input=static_input,
                    generated_ids=generated_ids
                ).item()
                
                if next_id == instance.char_eos.item():
                    break
                    
                generated_ids.append(next_id)

            return IdMapper.ids_to_str(generated_ids)

    def monitor(self, batch: Batch) -> None:
        instance = batch.get_random_instance()
        text_pred = self.predict_text(instance)
        instance.parsed.ink.visualise(name=text_pred)


if __name__ == "__main__":
    from core.data_schema.parsed import Parsed
    from core.utils.distributed_context import distributed_context
    from model.modules.embedder import VectorEmbedder

    parsed = Parsed.load_random()
    repr_tensor = torch.randn(100, 100).to(distributed_context.device)
    instance = Instance(parsed=parsed, _repr_tensor=torch.randn(100, 100))
    model = RecognitionModel(repr_embedder=VectorEmbedder(100)).to(distributed_context.device)
    model.eval()
    text_pred = model.predict_text(instance)
    print(text_pred)