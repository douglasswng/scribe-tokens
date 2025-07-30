from typing import Self
from enum import Enum
from dataclasses import dataclass

from core.repr import ReprId, TokenReprId


class Task(Enum):
    GENERATION = 'generation'
    PRETRAINING = 'pretraining'
    RECOGNITION_SFT = 'recognition_sft'
    POSTTRAINING = 'posttraining'


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
    def use_reference(self) -> bool:
        if self.task == Task.GENERATION:
            return True
        else:
            return False
        
    
if __name__ == "__main__":
    for model_id in ModelId.create_defaults():
        print(model_id)