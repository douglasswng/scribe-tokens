from typing import Self

import torch
from torch import Tensor
from pydantic import BaseModel

from core.repr import Repr, VectorReprId
from core.data_schema import DigitalInk, Stroke, Point
from core.utils import get_stroke_point_iterator
from core.constants import STD


class Point3(BaseModel):
    dx: float
    dy: float
    pen_up: bool

    def __str__(self) -> str:
        return f"({self.dx}, {self.dy}, {self.pen_up})"
    
    @classmethod
    def from_tensor(cls, tensor: Tensor) -> 'Point3':
        assert tensor.shape == (3,)
        assert tensor[2].item() in {0.0, 1.0}

        return cls(dx=tensor[0].item() * STD,
                   dy=tensor[1].item() * STD,
                   pen_up=tensor[2].item() == 1.0)
    
    def to_tensor(self) -> Tensor:
        return torch.tensor([self.dx / STD,
                             self.dy / STD,
                             int(self.pen_up)],
                            dtype=torch.float32)


class Point3Repr(Repr):
    def __init__(self, points: list[Point3]):
        self._points = points

    def __str__(self) -> str:
        return ' → '.join([str(point) for point in self._points])

    @classmethod
    def from_ink(cls, id: VectorReprId, ink: DigitalInk) -> Self:
        points: list[Point3] = []
        for point in get_stroke_point_iterator(ink).rel_points:
            points.append(Point3(dx=point.x, dy=point.y, pen_up=point.is_stroke_end))
        return cls(points)

    @classmethod
    def from_tensor(cls, id: VectorReprId, tensor: Tensor) -> Self:
        points: list[Point3] = []
        for point_tensor in tensor:
            points.append(Point3.from_tensor(point_tensor))
        return cls(points=points)

    def to_ink(self, id: VectorReprId) -> DigitalInk:
        strokes: list[Stroke] = []
        position = Point(x=0.0, y=0.0)
        stroke_points: list[Point] = [position]
        for point in self._points:
            position += Point(x=point.dx, y=point.dy)
            stroke_points.append(Point(x=position.x, y=position.y))
            if point.pen_up:
                strokes.append(Stroke(points=stroke_points))
                stroke_points = []
        return DigitalInk(strokes=strokes)
    
    def to_tensor(self, id: VectorReprId) -> Tensor:
        return torch.stack([point.to_tensor() for point in self._points], dim=0)