from typing import Self

import torch
from torch import Tensor
from pydantic import BaseModel, field_validator

from core.repr import Repr, VectorReprId
from core.data_schema import DigitalInk, Stroke, Point
from core.utils import get_stroke_point_iterator


class Point5(BaseModel):
    dx: float
    dy: float
    pen_up: bool
    pen_down: bool
    end: bool

    @field_validator('end')
    @classmethod
    def validate_pen_state(cls, v, info):
        states = info.data['pen_up'], info.data['pen_down'], v
        if sum(states) != 1:
            raise ValueError("Exactly one of pen_up, pen_down, and end must be True")
        return v

    def __str__(self) -> str:
        return f"({self.dx}, {self.dy}, {self.pen_up}, {self.pen_down}, {self.end})"
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'Point5':
        assert tensor.shape == (5,)
        assert torch.all((tensor[2:5] == 0.0) | (tensor[2:5] == 1.0))  # all pen states must be 0 or 1
        assert sum(tensor[2:5]) == 1.0  # exactly one pen state must be 1

        return cls(dx=tensor[0].item(),
                   dy=tensor[1].item(),
                   pen_up=(tensor[2].item() == 1.0),
                   pen_down=(tensor[3].item() == 1.0),
                   end=(tensor[4].item() == 1.0))
    
    @classmethod
    def get_start(cls) -> Self:  # to trigger generation
        return cls(dx=0.0, dy=0.0, pen_up=False, pen_down=True, end=False)
    
    def to_tensor(self) -> Tensor:
        pen_state = [int(self.pen_up), int(self.pen_down), int(self.end)]
        return torch.tensor([self.dx,
                             self.dy,
                             *pen_state],
                            dtype=torch.float32)
    

class Point5Repr(Repr):
    def __init__(self, points: list[Point5]):
        self._points = points

    def __str__(self) -> str:
        return ' → '.join([str(point) for point in self._points])

    @classmethod
    def from_ink(cls, id: VectorReprId, ink: DigitalInk) -> Self:
        points: list[Point5] = [Point5.get_start()]
        for point in get_stroke_point_iterator(ink).rel_points:
            pen_up, pen_down = point.is_stroke_end, not point.is_stroke_end
            points.append(Point5(dx=point.x, dy=point.y,
                                 pen_up=pen_up, pen_down=pen_down, end=False))
        if points:
            update = {'pen_up': False, 'pen_down': False, 'end': True}
            points[-1] = points[-1].model_copy(update=update)
        return cls(points)

    @classmethod
    def from_tensor(cls, id: VectorReprId, tensor: Tensor) -> Self:
        points: list[Point5] = []
        for point_tensor in tensor:
            points.append(Point5.from_tensor(point_tensor))
        return cls(points=points)

    def to_ink(self, id: VectorReprId) -> DigitalInk:
        strokes: list[Stroke] = []
        position = Point(x=0.0, y=0.0)
        stroke_points: list[Point] = [position]
        for point in self._points[1:]:  # skip the start point

            position += Point(x=point.dx, y=point.dy)
            stroke_points.append(Point(x=position.x, y=position.y))
            if point.pen_up or point.end:
                strokes.append(Stroke(points=stroke_points))
                stroke_points = []
            if point.end:
                break
        return DigitalInk(strokes=strokes)
    
    def to_tensor(self, id: VectorReprId) -> Tensor:
        return torch.stack([point.to_tensor() for point in self._points], dim=0)