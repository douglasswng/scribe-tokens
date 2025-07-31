import random
import numpy as np
import time

from core.data_schema import Parsed, DigitalInk
from core.constants import SCALE_RANGE, SHEAR_FACTOR, ROTATE_ANGLE, JITTER_SIGMA, AUGMENT_PROB


def scale_coords(coords: list[np.ndarray], scale_factor: float) -> list[np.ndarray]:
    """Apply scaling transformation to coordinates."""
    return [stroke_coords * scale_factor for stroke_coords in coords]


def shear_coords(coords: list[np.ndarray], shear_factor: float) -> list[np.ndarray]:
    """Apply shearing transformation to coordinates."""
    shear_matrix = np.array([[1, shear_factor], [0, 1]])
    return [stroke_coords @ shear_matrix.T for stroke_coords in coords]


def rotate_coords(coords: list[np.ndarray], angle_degrees: float) -> list[np.ndarray]:
    """Apply rotation transformation to coordinates."""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return [stroke_coords @ rotation_matrix.T for stroke_coords in coords]


def jitter_coords(coords: list[np.ndarray], sigma: float) -> list[np.ndarray]:
    """Apply jitter (random noise) to coordinates."""
    if sigma <= 0:
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
        start_time = time.perf_counter()
        coords = parsed.ink.to_coords()
        np_coords = [np.array(stroke) for stroke in coords]
        coord_conversion_time = time.perf_counter() - start_time
        
        # Apply transformations in sequence with timing
        start_time = time.perf_counter()
        np_coords = scale_coords(np_coords, cls._config.scale_factor)
        scale_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        np_coords = shear_coords(np_coords, cls._config.shear_factor)
        shear_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        np_coords = rotate_coords(np_coords, cls._config.rotate_angle)
        rotate_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        np_coords = jitter_coords(np_coords, cls._config.jitter_sigma)
        jitter_time = time.perf_counter() - start_time
        
        # Convert back to DigitalInk
        start_time = time.perf_counter()
        augmented_ink = DigitalInk.from_coords(np_coords)
        ink_conversion_time = time.perf_counter() - start_time
        
        # Print timing results
        total_time = coord_conversion_time + scale_time + shear_time + rotate_time + jitter_time + ink_conversion_time
        print(f"Augmentation timing:")
        print(f"  Coord conversion: {coord_conversion_time*1000:.3f}ms")
        print(f"  Scale:           {scale_time*1000:.3f}ms")
        print(f"  Shear:           {shear_time*1000:.3f}ms")
        print(f"  Rotate:          {rotate_time*1000:.3f}ms")
        print(f"  Jitter:          {jitter_time*1000:.3f}ms")
        print(f"  Ink conversion:  {ink_conversion_time*1000:.3f}ms")
        print(f"  Total:           {total_time*1000:.3f}ms")
        
        return parsed.model_copy(update={"ink": augmented_ink})
    
    @classmethod
    def reset_config(cls):
        cls._config = AugmenterConfig()


if __name__ == "__main__":
    parsed = Parsed.load_random()
    parsed.visualise()
    augmented_parsed = Augmenter.augment(parsed)
    augmented_parsed.visualise()