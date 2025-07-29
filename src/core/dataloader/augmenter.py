import random

from core.data_schema import Parsed, DigitalInk
from core.constants import SCALE_RANGE, SHEAR_FACTOR, ROTATE_ANGLE, JITTER_SIGMA, AUGMENT_PROB


class AugmenterConfig:
    def __init__(self):
        self.scale_factor = 1 + self._sample_arg(1 - SCALE_RANGE, 1 + SCALE_RANGE)
        self.shear_factor = self._sample_arg(-SHEAR_FACTOR, SHEAR_FACTOR)
        self.rotate_angle = self._sample_arg(-ROTATE_ANGLE, ROTATE_ANGLE)
        self.jitter_sigma = self._sample_arg(0, JITTER_SIGMA)

        self.jitter_sigma = JITTER_SIGMA

    def _sample_arg(self, min_val: float, max_val: float, default: float=0) -> float:
        return random.uniform(min_val, max_val)  # debug
        if random.random() <= AUGMENT_PROB:
            return random.uniform(min_val, max_val)
        return default


class Augmenter:
    _config: AugmenterConfig = AugmenterConfig()

    @classmethod
    def _augment_ink(cls, ink: DigitalInk, config: AugmenterConfig) -> DigitalInk:
        # ink = ink.scale(config.scale_factor)
        # ink = ink.shear(config.shear_factor)
        # ink = ink.rotate(config.rotate_angle)
        ink = ink.jitter(config.jitter_sigma)
        return ink

    @classmethod
    def augment(cls, parsed: Parsed) -> Parsed:
        augmented_ink = cls._augment_ink(parsed.ink, cls._config)
        return parsed.model_copy(update={"ink": augmented_ink})
    
    @classmethod
    def reset_config(cls):
        cls._config = AugmenterConfig()


if __name__ == "__main__":
    parsed = Parsed.load_random()
    parsed.visualise()
    augmented_parsed = Augmenter.augment(parsed)
    augmented_parsed.visualise()