from typing import Self
from enum import Enum
from dataclasses import dataclass

from core.repr import ReprId, TokenReprId, VectorReprId


class Task(Enum):
    RECOGNITION = 'recognition'
    GENERATION = 'generation'
    PRETRAINING_NTP = 'pretraining_ntp'
    RECOGNITION_SFT = 'recognition_sft'
    GENERATION_SFT = 'generation_sft'


@dataclass(frozen=True)
class ModelId:
    task: Task
    repr_id: ReprId

    def __str__(self) -> str:
        return f"Task: {self.task.value}, Repr: {self.repr_id}"
    
    @classmethod
    def _get_repr_ids(cls, task: Task) -> list[ReprId]:
        token_repr_ids: list[ReprId] = TokenReprId.create_defaults()
        vector_repr_id: ReprId = VectorReprId.create_point5()
        return token_repr_ids + [vector_repr_id]
    
    @classmethod
    def create_task_model_ids(cls, task: Task) -> list[Self]:
        return [cls(task=task, repr_id=repr_id) for repr_id in cls._get_repr_ids(task)]

    @classmethod
    def create_defaults(cls) -> list[Self]:
        model_ids = []
        for task in Task:
            model_ids.extend(cls.create_task_model_ids(task))
        return model_ids
        
    
if __name__ == "__main__":
    for model_id in ModelId.create_defaults():
        print(model_id)