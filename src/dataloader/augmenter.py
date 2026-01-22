import random

import numpy as np
from core.constants import AUGMENT_PROB, JITTER_SIGMA, ROTATE_ANGLE, SCALE_RANGE, SHEAR_FACTOR

from core.data_schema import DigitalInk, Parsed


def scale_coords(coords: list[np.ndarray], scale_factor: float) -> list[np.ndarray]:
    """Apply scaling transformation to coordinates."""
    if scale_factor == 1:
        return coords
    return [stroke_coords * scale_factor for stroke_coords in coords]


def shear_coords(coords: list[np.ndarray], shear_factor: float) -> list[np.ndarray]:
    """Apply shearing transformation to coordinates."""
    if shear_factor == 0:
        return coords
    shear_matrix = np.array([[1, shear_factor], [0, 1]])
    return [stroke_coords @ shear_matrix.T for stroke_coords in coords]


def rotate_coords(coords: list[np.ndarray], angle_degrees: float) -> list[np.ndarray]:
    """Apply rotation transformation to coordinates."""
    if angle_degrees == 0:
        return coords
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return [stroke_coords @ rotation_matrix.T for stroke_coords in coords]


def jitter_coords(coords: list[np.ndarray], sigma: float) -> list[np.ndarray]:
    """Apply jitter (random noise) to coordinates."""
    if sigma == 0:
        return coords

    jittered_coords = []
    for stroke_coords in coords:
        if len(stroke_coords) > 0:
            jitter = np.random.normal(0, sigma, stroke_coords.shape)
            jittered_coords.append(stroke_coords + jitter)
        else:
            jittered_coords.append(stroke_coords.copy())
    return jittered_coords


def reverse_coords(coords: list[np.ndarray]) -> list[np.ndarray]:
    """Reverse the strokes and points within each stroke."""
    return [stroke_coords[::-1] for stroke_coords in reversed(coords)]


class AugmenterConfig:
    def __init__(self):
        self.scale_factor = 1 + self._sample_arg(SCALE_RANGE, -SCALE_RANGE)
        self.shear_factor = self._sample_arg(SHEAR_FACTOR, -SHEAR_FACTOR)
        self.rotate_angle = self._sample_arg(ROTATE_ANGLE, -ROTATE_ANGLE)
        self.jitter_sigma = self._sample_arg(JITTER_SIGMA)

        self.reverse = random.random() <= AUGMENT_PROB

    def _sample_arg(self, max_val: float, min_val: float = 0, default: float = 0) -> float:
        if random.random() <= AUGMENT_PROB:
            return random.uniform(min_val, max_val)
        return default


class Augmenter:
    _config: AugmenterConfig = AugmenterConfig()

    @classmethod
    def augment(cls, parsed: Parsed) -> Parsed:
        # Convert to coordinates
        coords = parsed.ink.to_coords()
        np_coords = [np.array(stroke) for stroke in coords]

        # Apply transformations in sequence
        np_coords = scale_coords(np_coords, cls._config.scale_factor)
        np_coords = shear_coords(np_coords, cls._config.shear_factor)
        np_coords = rotate_coords(np_coords, cls._config.rotate_angle)
        np_coords = jitter_coords(np_coords, cls._config.jitter_sigma)

        # if cls._config.reverse:
        #     np_coords = reverse_coords(np_coords)

        # Convert back to DigitalInk
        augmented_ink = DigitalInk.from_coords(np_coords)
        return parsed.model_copy(update={"ink": augmented_ink})

    @classmethod
    def reset_config(cls):
        cls._config = AugmenterConfig()


if __name__ == "__main__":
    from core.repr import TokenReprId
    from repr.factory import DefaultReprFactory

    parsed = Parsed.load_random()
    parsed.visualise()
    augmented_parsed = Augmenter.augment(parsed)
    augmented_parsed.visualise()

    repr_id = TokenReprId.create_scribe()

    original_tensor = DefaultReprFactory.ink_to_tensor(repr_id, parsed.ink)
    augmented_tensor = DefaultReprFactory.ink_to_tensor(repr_id, augmented_parsed.ink)

    original_ink = DefaultReprFactory.tensor_to_ink(repr_id, original_tensor)
    augmented_ink = DefaultReprFactory.tensor_to_ink(repr_id, augmented_tensor)
    original_ink.visualise()
    augmented_ink.visualise()

    print("Ink length:", len(parsed.ink))
    print("Original token count:", original_tensor.size(0))
    print("Augmented token count:", augmented_tensor.size(0))
