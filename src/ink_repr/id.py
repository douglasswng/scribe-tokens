from enum import StrEnum
from typing import Self

from ink_tokeniser.id import TokeniserId


class VectorType(StrEnum):
    POINT3 = "Point-3"
    POINT5 = "Point-5"


class VectorReprId:
    def __init__(self, type: VectorType):
        self.type = type

    def __str__(self) -> str:
        return self.type

    @classmethod
    def create_point5(cls) -> Self:
        return cls(VectorType.POINT5)

    # @property
    # def is_vector(self) -> bool:
    #     return True

    # @property
    # def has_oov(self) -> bool:
    #     return False

    # @property
    # def dim(self) -> int:
    #     match self.type:
    #         case VectorType.POINT3:
    #             return 3
    #         case VectorType.POINT5:
    #             return 5
    #         case _:
    #             raise ValueError(f"Unknown vector repr type: {self.type}")


ReprId = VectorReprId | TokeniserId
