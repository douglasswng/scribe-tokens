from dataclasses import dataclass

from torch import Tensor

from core.data_schema.parsed import Parsed


@dataclass(frozen=True)
class Instance:
    parsed: Parsed
    repr_tensor: Tensor  # [repr_len] / [repr_len, repr_dim] for token / vector repr
    writer_id_tensor: Tensor  # [1]
    char_ids_tensor: Tensor  # [char_len]