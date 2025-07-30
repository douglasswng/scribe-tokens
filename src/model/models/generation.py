import torch
from torch import Tensor

from core.model import LocalModel, ModelId
from core.data_schema import Batch, DigitalInk
from repr.factory import DefaultReprFactory
from model.modules.embedder import Embedder, CharEmbedder, TokenEmbedder
from model.modules.encoder import Encoder
from model.modules.decoder import Decoder
from model.models.gen_mixin import GenMixin, MDNParam


class GenerationModel(LocalModel, GenMixin):
    def __init__(self, model_id: ModelId):
        super().__init__()
        self._model_id = model_id

        self._repr_embedder: Embedder = TokenEmbedder()
        self._char_embedder: Embedder = CharEmbedder()
        self._repr_decoder = Decoder()

    def _forward(self, ref_repr: Tensor,
                 main_char: Tensor,
                 main_repr_input: Tensor,
                 ref_repr_pad: Tensor | None = None,
                 main_char_pad: Tensor | None = None,
                 main_repr_input_pad: Tensor | None = None
                 ) -> Tensor | MDNParam:
        ref_repr = self._repr_embedder.embed(ref_repr)
        main_char = self._char_embedder.embed(main_char)
        main_repr_input = self._repr_embedder.embed(main_repr_input)
        
        ref_repr = self._ref_repr_encoder.forward(x=ref_repr, pad=ref_repr_pad)
        main_char = self._main_char_decoder.forward(tgt=main_char,
                                                    memory=ref_repr,
                                                    tgt_pad=main_char_pad,
                                                    memory_pad=ref_repr_pad)
        main_repr_input = self._main_repr_decoder.forward(tgt=main_repr_input,
                                                          memory=main_char,
                                                          tgt_pad=main_repr_input_pad,
                                                          memory_pad=main_char_pad)

        main_repr_pred = self._repr_embedder.unembed(main_repr_input)
        return main_repr_pred

    def losses(self, batch: Batch) -> dict[str, Tensor]:
        if batch.reference_batch is None:
            raise ValueError("Reference batch is required for generation model")
        
        main_batch, ref_batch = batch.main_batch, batch.reference_batch
        main_repr_pred = self._forward(ref_repr=ref_batch.repr, 
                                       main_char=main_batch.char, 
                                       main_repr_input=main_batch.repr_input, 
                                       ref_repr_pad=ref_batch.repr_pad,
                                       main_char_pad=main_batch.char_pad,
                                       main_repr_input_pad=main_batch.repr_input_pad)

        loss = self._loss(main_repr_pred, main_batch.repr_target, main_batch.repr_target_pad)
        loss_name = 'ntp_ce' if self._is_token_repr(main_batch.repr_target) else 'ntp_nll'
        return {loss_name: loss}
    
    def _get_last_pred(self, main_repr_pred: Tensor | MDNParam) -> Tensor | MDNParam:
        if self._is_token_pred(main_repr_pred):
            return main_repr_pred[:, -1]
        elif self._is_vector_pred(main_repr_pred):
            last_pred = tuple(tensor[:, -1] for tensor in main_repr_pred)
            assert len(last_pred) == 5
            return last_pred
        else:
            raise ValueError(f"Unsupported prediction {main_repr_pred}")
    
    def _generate_next_tensor(self, ref_repr: Tensor,
                              main_char: Tensor,
                              gen_repr: Tensor,
                              ref_repr_pad: Tensor,
                              main_char_pad: Tensor) -> Tensor:
        gen_repr_pred = self._forward(ref_repr=ref_repr,
                                      main_char=main_char,
                                      main_repr_input=gen_repr,
                                      ref_repr_pad=ref_repr_pad,
                                      main_char_pad=main_char_pad)
        last_pred = self._get_last_pred(gen_repr_pred)
        next_tensor = self._sample_next_tensor(last_pred)
        return next_tensor
    
    def _generate_tensor(self, ref_repr: Tensor, main_char: Tensor,
                         ref_repr_pad: Tensor,
                         main_char_pad: Tensor) -> Tensor:
        gen_repr = self._get_start(ref_repr)
        while not self._is_end(gen_repr):
            next_tensor = self._generate_next_tensor(ref_repr, main_char, gen_repr,
                                                     ref_repr_pad, main_char_pad)
            gen_repr = torch.cat([gen_repr, next_tensor], dim=1)
        return gen_repr
    
    def batch_generate_ink(self, ref_repr: Tensor, main_char: Tensor, 
                           ref_repr_pad: Tensor,
                           main_char_pad: Tensor) -> list[DigitalInk]:
        if self.training:
            raise ValueError("Generation is not supported in training mode")

        with torch.no_grad():
            tensor = self._generate_tensor(ref_repr, main_char, ref_repr_pad, main_char_pad)
            inks = [DefaultReprFactory.tensor_to_ink(id=self._model_id.repr_id, tensor=instance_tensor)
                    for instance_tensor in tensor]
            return inks
    
    def monitor(self, batch: Batch) -> None:
        sample = batch.get_random_sample()
        main_batch, ref_batch = sample.main_batch, sample.reference_batch

        if ref_batch is None:
            raise ValueError("Reference batch is required for generation model")
        
        ink = self.batch_generate_ink(ref_batch.repr, main_batch.char,
                                      ref_batch.repr_pad, main_batch.char_pad)[0]
        ink.visualise(name=main_batch.instances[0].parsed.text)

        
if __name__ == "__main__":
    from core.model import Task
    from dataloader.create import create_dataloaders
    from model.factory import ReprEmbedderFactory
    from core.utils import distributed_context

    for model_id in ModelId.create_task_model_ids(Task.GENERATION)[:]:
        print(model_id)
        train_loader, val_loader, test_loader = create_dataloaders(
            model_id=model_id,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        repr_embedder = ReprEmbedderFactory.create(model_id)
        model = GenerationModel(repr_embedder, model_id).to(distributed_context.device)
        for batch in train_loader:
            model.monitor(batch)
            break