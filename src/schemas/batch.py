import random
from dataclasses import dataclass

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
