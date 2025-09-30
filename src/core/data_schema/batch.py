import random
from dataclasses import dataclass

from core.data_schema.instance import Instance


@dataclass(frozen=True)
class SingletonBatch:
    instances: list[Instance]

    def get_random_instance(self) -> Instance:
        random_instance = random.choice(self.instances)
        return random_instance


@dataclass(frozen=True)
class PairBatch:
    main_instances: list[Instance]
    ref_instances: list[Instance]

    def get_random_instance_pair(self) -> tuple[Instance, Instance]:
        random_idx = random.randint(0, len(self.main_instances) - 1)
        main_instance = self.main_instances[random_idx]
        ref_instance = self.ref_instances[random_idx]
        return main_instance, ref_instance


type Batch = SingletonBatch | PairBatch
