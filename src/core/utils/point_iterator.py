from pydantic import BaseModel
from core.data_schema import DigitalInk, Point


class StrokePoint[T: (float, int)](BaseModel):
    x: T
    y: T
    is_stroke_start: bool = False
    is_stroke_end: bool = False

    def __sub__(self, other: 'StrokePoint') -> 'StrokePoint':
        return self.model_copy(update={'x': self.x - other.x, 'y': self.y - other.y})
    
    @property
    def point(self) -> Point[T]:
        return Point(x=self.x, y=self.y)


class StrokePointIterator[T: (float, int)](BaseModel):  # cleaner way to parse into different representations
    digital_ink: DigitalInk[T]

    @classmethod
    def from_digital_ink(cls, digital_ink: DigitalInk[T]) -> 'StrokePointIterator[T]':
        return cls(digital_ink=digital_ink)
    
    @property
    def abs_points(self) -> list[StrokePoint[T]]:
        points: list[StrokePoint[T]] = []
        for stroke in self.digital_ink.strokes:
            for i, point in enumerate(stroke.points):
                stroke_point = StrokePoint(x=point.x, y=point.y)
                if i == 0:
                    stroke_point.is_stroke_start = True
                if i == len(stroke.points) - 1:
                    stroke_point.is_stroke_end = True
                points.append(stroke_point)
        return points

    @property
    def rel_points(self) -> list[StrokePoint[T]]:
        abs_points = self.abs_points
        points: list[StrokePoint[T]] = []
        for p1, p2 in zip(abs_points, abs_points[1:]):
            points.append(p2 - p1)
        return points
    

def get_stroke_point_iterator(digital_ink: DigitalInk[int]) -> StrokePointIterator[int]:
    return StrokePointIterator.from_digital_ink(digital_ink)
    
    
if __name__ == '__main__':
    digital_ink = DigitalInk.load_test()
    stroke_point_iterator = StrokePointIterator.from_digital_ink(digital_ink)
    print(stroke_point_iterator.abs_points)
    print(stroke_point_iterator.rel_points)