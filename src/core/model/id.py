from typing import Self, Literal
from enum import Enum
from dataclasses import dataclass

from core.repr import ReprId, TokenReprId


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
    def create_task_defaults(cls, task: Task) -> list[Self]:
        return [cls(task=task, repr_id=TokenReprId.create_scribe())]

    @classmethod
    def create_defaults(cls) -> list[Self]:
        model_ids = []
        for task in Task:
            model_id = cls(task=task, repr_id=TokenReprId.create_scribe())
            model_ids.append(model_id)
        return model_ids
    
    @property
    def context_type(self) -> Literal['repr', 'char', None]:
        match self.task:
            case Task.RECOGNITION | Task.RECOGNITION_SFT:
                return 'repr'
            case Task.GENERATION | Task.GENERATION_SFT:
                return 'char'
            case Task.PRETRAINING_NTP:
                return None
    
    @property
    def main_type(self) -> Literal['repr', 'char']:
        match self.task:
            case Task.RECOGNITION | Task.RECOGNITION_SFT:
                return 'char'
            case Task.GENERATION | Task.GENERATION_SFT | Task.PRETRAINING_NTP:
                return 'repr'
    
if __name__ == "__main__":
    for model_id in ModelId.create_defaults():
        print(model_id)