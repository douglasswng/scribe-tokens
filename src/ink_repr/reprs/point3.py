from typing import Self

import torch
from pydantic import BaseModel
from torch import Tensor

from constants import INK_SCALE
from ink_repr.repr import InkRepr
from schemas.ink import DigitalInk, Point, Stroke
from utils.point_iterator import get_stroke_point_iterator


class Point3(BaseModel):
    dx: float
    dy: float
    pen_up: bool

    def __str__(self) -> str:
        return f"({self.dx}, {self.dy}, {self.pen_up})"

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> "Point3":
        assert tensor.shape == (3,)
        assert tensor[2].item() in {0.0, 1.0}

        return cls(dx=tensor[0].item(), dy=tensor[1].item(), pen_up=tensor[2].item() == 1.0)

    def to_tensor(self) -> Tensor:
        return torch.tensor([self.dx, self.dy, int(self.pen_up)], dtype=torch.float32)


class Point3Repr(InkRepr):
    def __init__(self, points: list[Point3]):
        self._points = points

    def __str__(self) -> str:
        return " → ".join([str(point) for point in self._points])

    @classmethod
    def from_ink(cls, ink: DigitalInk) -> Self:
        points: list[Point3] = []
        for point in get_stroke_point_iterator(ink).rel_points:
            points.append(Point3(dx=point.x, dy=point.y, pen_up=point.is_stroke_end))
        return cls(points)

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> Self:
        # Only scale coordinates, not pen state flags
        scaled_tensor = tensor.clone()
        scaled_tensor[:, :2] = scaled_tensor[:, :2] / INK_SCALE
        points: list[Point3] = []
        for point_tensor in scaled_tensor:
            points.append(Point3.from_tensor(point_tensor))
        return cls(points=points)

    def to_ink(self) -> DigitalInk:
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

    def to_tensor(self) -> Tensor:
        tensor = torch.stack([point.to_tensor() for point in self._points], dim=0)
        # Only scale coordinates, not pen state flags
        tensor[:, :2] = tensor[:, :2] * INK_SCALE
        return tensor
