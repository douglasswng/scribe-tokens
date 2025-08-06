from dataclasses import dataclass
import random

from core.data_schema.instance import Instance


@dataclass(frozen=True)
class Batch:
    instances: list[Instance]

    def get_random_instance(self) -> Instance:
        random_instance = random.choice(self.instances)
        return random_instance