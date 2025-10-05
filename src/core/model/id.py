from typing import Self
from enum import Enum
from dataclasses import dataclass

from core.repr import ReprId, TokenReprId, VectorReprId
from core.constants import (
    GENERATION_NUM_EPOCHS, GENERATION_LEARNING_RATE,
    RECOGNITION_NUM_EPOCHS, RECOGNITION_LEARNING_RATE,
    PRETRAINING_NUM_EPOCHS, PRETRAINING_LEARNING_RATE,
    RECOGNITION_SFT_NUM_EPOCHS, RECOGNITION_SFT_LEARNING_RATE,
    GENERATION_SFT_NUM_EPOCHS, GENERATION_SFT_LEARNING_RATE
)


class Task(Enum):
    GENERATION = 'generation'
    ENHANCEMENT = 'enhancement'
    RECOGNITION = 'recognition'
    PRETRAINING_NTP = 'pretraining_ntp'
    RECOGNITION_SFT = 'recognition_sft'
    GENERATION_SFT = 'generation_sft'

    @property
    def is_sft(self) -> bool:
        return self in {Task.RECOGNITION_SFT, Task.GENERATION_SFT}

    @property
    def use_reference(self) -> bool:
        return self in {Task.GENERATION, Task.GENERATION_SFT}

    @property
    def num_epochs(self) -> int:
        match self:
            case Task.GENERATION:
                return GENERATION_NUM_EPOCHS
            case Task.ENHANCEMENT:
                return GENERATION_NUM_EPOCHS  # TODO: fix
            case Task.RECOGNITION:
                return RECOGNITION_NUM_EPOCHS
            case Task.PRETRAINING_NTP:
                return PRETRAINING_NUM_EPOCHS
            case Task.RECOGNITION_SFT:
                return RECOGNITION_SFT_NUM_EPOCHS
            case Task.GENERATION_SFT:
                return GENERATION_SFT_NUM_EPOCHS
            case _:
                raise ValueError(f"Invalid task: {self}")

    @property
    def learning_rate(self) -> float:
        match self:
            case Task.GENERATION:
                return GENERATION_LEARNING_RATE
            case Task.ENHANCEMENT:
                return GENERATION_LEARNING_RATE  # TODO: fix
            case Task.RECOGNITION:
                return RECOGNITION_LEARNING_RATE
            case Task.PRETRAINING_NTP:
                return PRETRAINING_LEARNING_RATE
            case Task.RECOGNITION_SFT:
                return RECOGNITION_SFT_LEARNING_RATE
            case Task.GENERATION_SFT:
                return GENERATION_SFT_LEARNING_RATE
            case _:
                raise ValueError(f"Invalid task: {self}")


@dataclass(frozen=True)
class ModelId:
    task: Task
    repr_id: ReprId

    def __str__(self) -> str:
        return f"Task: {self.task.value}, Repr: {self.repr_id}"
    
    @classmethod
    def _get_repr_ids(cls) -> list[ReprId]:
        token_repr_ids: list[ReprId] = TokenReprId.create_defaults()
        vector_repr_id: ReprId = VectorReprId.create_point5()
        return  token_repr_ids + [vector_repr_id]
    
    @classmethod
    def create_task_model_ids(cls, task: Task) -> list[Self]:
        return [cls(task=task, repr_id=repr_id) for repr_id in cls._get_repr_ids()]

    @classmethod
    def create_defaults(cls) -> list[Self]:
        model_ids = []
        for task in Task:
            model_ids.extend(cls.create_task_model_ids(task))
        return model_ids
        
    
if __name__ == "__main__":
    for model_id in ModelId.create_defaults():
        print(model_id)