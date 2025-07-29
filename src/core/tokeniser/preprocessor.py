from typing import Protocol

from core.data_schema import DigitalInk


class Preprocessor(Protocol):
    def preprocess(self, ink: DigitalInk) -> DigitalInk[int]: ...
    def postprocess(self, ink: DigitalInk[int]) -> DigitalInk: ...


class DeltaPreprocessor(Preprocessor):
    def __init__(self, delta: int | float):
        self._delta = delta

    def preprocess(self, digital_ink: DigitalInk) -> DigitalInk[int]:
        return digital_ink.scale(1 / self._delta).discretise()
    
    def postprocess(self, digital_ink: DigitalInk[int]) -> DigitalInk:
        return digital_ink.scale(self._delta)
    

class DeltaSmoothPreprocessor(Preprocessor):
    def __init__(self, delta: int | float,
                 downsample_factor: int,
                 smooth_window_length: int=5,
                 smooth_polyorder: int=3):
        self._delta = delta
        self._downsample_factor = downsample_factor
        self._smooth_window_length = smooth_window_length
        self._smooth_polyorder = smooth_polyorder

    def preprocess(self, digital_ink: DigitalInk) -> DigitalInk[int]:
        return digital_ink.scale(1 / self._delta).discretise()
    
    def postprocess(self, digital_ink: DigitalInk[int]) -> DigitalInk:
        digital_ink = digital_ink.scale(self._delta)
        digital_ink = digital_ink.downsample(factor=self._downsample_factor)
        digital_ink = digital_ink.smooth(window_length=self._smooth_window_length, polyorder=self._smooth_polyorder)
        return digital_ink