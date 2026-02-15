import random
from dataclasses import dataclass
from typing import Self

from schemas.instance import Instance


@dataclass(frozen=True)
class Batch:
    instances: list[Instance]

    @property
    def size(self) -> int:
        return len(self.instances)

    def get_random_instance(self) -> Instance:
        random_instance = random.choice(self.instances)
        return random_instance

    def to_device(self) -> Self:
        moved_instances = [instance.to_device() for instance in self.instances]
        return type(self)(instances=moved_instances)
