import torch
from torch import Tensor

from core.data_schema import IdMapper
from core.data_schema import DigitalInk
from core.repr import TokenReprId
from core.utils import distributed_context
from repr.factory import DefaultReprFactory


class InkTextConverter:
    @classmethod
    def ink_to_tensor(cls, ink: DigitalInk) -> Tensor:
        repr_tensor = DefaultReprFactory.ink_to_tensor(id=TokenReprId.create_scribe(), ink=ink)
        repr_tensor = repr_tensor
        repr_tensor = repr_tensor.to(distributed_context.device)
        return repr_tensor

    @classmethod
    def text_to_tensor(cls, text: str) -> Tensor:
        char_tensor = torch.tensor(IdMapper.str_to_ids(text))
        char_tensor = char_tensor.to(distributed_context.device)
        char_tensor = char_tensor
        return char_tensor

    @classmethod
    def inks_to_tensors(cls, inks: list[DigitalInk]) -> tuple[Tensor, Tensor]:
        repr_tensors = [cls.ink_to_tensor(ink) for ink in inks]
        repr_tensor = torch.nn.utils.rnn.pad_sequence(repr_tensors, batch_first=True, padding_value=0)
        repr_pad = torch.zeros(repr_tensor.shape, dtype=torch.bool, device=repr_tensor.device)
        for i, tensor in enumerate(repr_tensors):
            repr_pad[i, :tensor.size(0)] = True
        return repr_tensor, repr_pad
    
    @classmethod
    def texts_to_tensors(cls, texts: list[str]) -> tuple[Tensor, Tensor]:
        char_tensors = [cls.text_to_tensor(text) for text in texts]
        char_tensor = torch.nn.utils.rnn.pad_sequence(char_tensors, batch_first=True, padding_value=0)
        char_pad = torch.zeros(char_tensor.shape, dtype=torch.bool, device=char_tensor.device)
        for i, tensor in enumerate(char_tensors):
            char_pad[i, :tensor.size(0)] = True
        return char_tensor, char_pad