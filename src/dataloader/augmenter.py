import random
import numpy as np

from core.data_schema import Parsed, DigitalInk
from core.constants import SCALE_RANGE, SHEAR_FACTOR, ROTATE_ANGLE, JITTER_SIGMA, AUGMENT_PROB


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


class AugmenterConfig:
    def __init__(self):
        self.scale_factor = 1 + self._sample_arg(-SCALE_RANGE, SCALE_RANGE)
        self.shear_factor = self._sample_arg(-SHEAR_FACTOR, SHEAR_FACTOR)
        self.rotate_angle = self._sample_arg(-ROTATE_ANGLE, ROTATE_ANGLE)
        self.jitter_sigma = self._sample_arg(0, JITTER_SIGMA)

    def _sample_arg(self, min_val: float, max_val: float, default: float=0) -> float:
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
        
        # Convert back to DigitalInk
        augmented_ink = DigitalInk.from_coords(np_coords)
        return parsed.model_copy(update={"ink": augmented_ink})
    
    @classmethod
    def reset_config(cls):
        cls._config = AugmenterConfig()


if __name__ == "__main__":
    parsed = Parsed.load_random()
    parsed.visualise()
    augmented_parsed = Augmenter.augment(parsed)
    augmented_parsed.visualise()