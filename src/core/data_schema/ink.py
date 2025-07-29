from typing import overload
import math
import re

from pydantic import BaseModel
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import numpy as np

from core.constants import TMP_DIR


def _reconstruct_point(x, y):
    return Point(x=x, y=y)

def _reconstruct_stroke(points):
    return Stroke(points=points)

def _reconstruct_digital_ink(strokes):
    return DigitalInk(strokes=strokes)


class Point[T: (float, int)](BaseModel):
    x: T
    y: T

    def __reduce_ex__(self, protocol):  # for pickling
        return (_reconstruct_point, (self.x, self.y))

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __add__(self, other: 'Point[T]') -> 'Point[T]':
        return Point(x=self.x + other.x, y=self.y + other.y)
    
    def __sub__(self, other: 'Point[T]') -> 'Point[T]':
        return Point(x=self.x - other.x, y=self.y - other.y)
    
    def __neg__(self) -> 'Point[T]':
        return Point(x=-self.x, y=-self.y)
    
    def __mul__(self, other: float) -> 'Point':
        return Point(x=self.x * other, y=self.y * other)
    
    def shear(self, shear_factor: float) -> 'Point':
        return Point(x=self.x + shear_factor * self.y, y=self.y)
    
    def rotate(self, angle_degrees: float) -> 'Point':
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        new_x = self.x * cos_a - self.y * sin_a
        new_y = self.x * sin_a + self.y * cos_a
        
        return Point(x=new_x, y=new_y)
    
    def round(self) -> 'Point[int]':
        return Point(x=round(self.x), y=round(self.y))


class Stroke[T: (float, int)](BaseModel):
    points: list[Point[T]]

    def __reduce_ex__(self, protocol):
        return (_reconstruct_stroke, (self.points,))

    def __len__(self) -> int:
        return len(self.points)
    
    def __str__(self) -> str:
        points_str = " → ".join(str(point) for point in self.points)
        return points_str

    def shift(self, shift: Point) -> 'Stroke':
        return Stroke(points=[point + shift for point in self.points])
    
    def scale(self, scale: float) -> 'Stroke':
        return Stroke(points=[point * scale for point in self.points])

    def scale_x(self, scale: float) -> 'Stroke':
        return Stroke(points=[Point(x=point.x * scale, y=point.y) for point in self.points])
    
    def scale_y(self, scale: float) -> 'Stroke':
        return Stroke(points=[Point(x=point.x, y=point.y * scale) for point in self.points])
    
    def shear(self, shear_factor: float) -> 'Stroke':
        return Stroke(points=[point.shear(shear_factor) for point in self.points])
    
    def rotate(self, angle_degrees: float) -> 'Stroke':
        return Stroke(points=[point.rotate(angle_degrees) for point in self.points])
    
    def discretise(self) -> 'Stroke[int]':
        return Stroke(points=[point.round() for point in self.points])

    def downsample(self, factor: int) -> 'Stroke':
        if len(self.points) <= 2:
            return Stroke(points=self.points[:])
        
        downsampled = [self.points[0]]
        
        for i in range(factor, len(self.points) - 1, factor):
            downsampled.append(self.points[i])
        
        downsampled.append(self.points[-1])
        
        return Stroke(points=downsampled)

    def smooth(self, window_length: int=5, polyorder: int=3) -> 'Stroke':
        if len(self.points) < 5:  # Savgol filter needs at least 5 points for window_length=5
            return Stroke(points=self.points[:])  # Return copy of original stroke
        
        # Extract x and y coordinates
        x_coords = np.array([point.x for point in self.points])
        y_coords = np.array([point.y for point in self.points])
        
        
        smoothed_x = savgol_filter(x_coords, window_length, polyorder)
        smoothed_y = savgol_filter(y_coords, window_length, polyorder)
        
        # Create new points with smoothed coordinates
        smoothed_points = [Point(x=float(x), y=float(y)) for x, y in zip(smoothed_x, smoothed_y)]
        
        return Stroke(points=smoothed_points)


class DigitalInk[T: (float, int)](BaseModel):
    strokes: list[Stroke[T]]

    def __reduce_ex__(self, protocol):
        return (_reconstruct_digital_ink, (self.strokes,))

    def __str__(self) -> str:
        line = '-' * 100
        strokes_str = f"{line}\nDigitalInk:\n"
        strokes_str += "\n\n".join(f"  stroke{i+1}: {stroke}" for i, stroke in enumerate(self.strokes))
        strokes_str += f"\n{line}"
        return strokes_str
    
    def __len__(self) -> int:
        return sum(len(stroke) for stroke in self.strokes)

    @overload
    @classmethod
    def from_coords(cls, raw_strokes: list[list[tuple[int, int]]], to_origin: bool=True) -> 'DigitalInk[int]': ...
    
    @overload
    @classmethod
    def from_coords(cls, raw_strokes: list[list[tuple[float, float]]], to_origin: bool=True) -> 'DigitalInk[float]': ...

    @classmethod
    def from_coords(cls, raw_strokes, to_origin: bool=True):
        strokes = []
        for stroke in raw_strokes:
            points = [Point(x=coord[0], y=coord[1]) for coord in stroke]
            strokes.append(Stroke(points=points))
        digital_ink = DigitalInk(strokes=strokes)
        if to_origin:
            digital_ink = digital_ink.to_origin()
        return digital_ink
    
    @classmethod
    def load_test(cls) -> 'DigitalInk[int]':
        raw_strokes = [
            [(0, 0), (1, 0)],
            [(2, 1), (4, -1)],
        ]
        return cls.from_coords(raw_strokes)

    @property
    def bbox(self) -> tuple[Point, Point]:
        min_x = min(point.x for stroke in self.strokes for point in stroke.points)
        min_y = min(point.y for stroke in self.strokes for point in stroke.points)
        max_x = max(point.x for stroke in self.strokes for point in stroke.points)
        max_y = max(point.y for stroke in self.strokes for point in stroke.points)
        return Point(x=min_x, y=min_y), Point(x=max_x, y=max_y)

    @property
    def height(self) -> float:
        top_left, bottom_right = self.bbox
        return bottom_right.y - top_left.y

    @property
    def width(self) -> float:
        top_left, bottom_right = self.bbox
        return bottom_right.x - top_left.x

    @property
    def start(self) -> Point:
        return self.strokes[0].points[0]

    def to_origin(self) -> 'DigitalInk':
        top_left, _ = self.bbox
        return self.shift(-top_left)
    
    def shift(self, shift: Point) -> 'DigitalInk':
        return DigitalInk(strokes=[stroke.shift(shift) for stroke in self.strokes])
    
    def scale(self, scale: float) -> 'DigitalInk':
        return DigitalInk(strokes=[stroke.scale(scale) for stroke in self.strokes])

    def scale_x(self, scale: float) -> 'DigitalInk':
        return DigitalInk(strokes=[stroke.scale_x(scale) for stroke in self.strokes])

    def scale_y(self, scale: float) -> 'DigitalInk':
        return DigitalInk(strokes=[stroke.scale_y(scale) for stroke in self.strokes])
    
    def shear(self, shear_factor: float) -> 'DigitalInk':
        return DigitalInk(strokes=[stroke.shear(shear_factor) for stroke in self.strokes])
    
    def rotate(self, angle_degrees: float) -> 'DigitalInk':
        return DigitalInk(strokes=[stroke.rotate(angle_degrees) for stroke in self.strokes])
    
    def discretise(self) -> 'DigitalInk[int]':
        return DigitalInk(strokes=[stroke.discretise() for stroke in self.strokes])

    def downsample(self, factor: int) -> 'DigitalInk':
        return DigitalInk(strokes=[stroke.downsample(factor) for stroke in self.strokes])

    def smooth(self, window_length: int=5, polyorder: int=3) -> 'DigitalInk':
        return DigitalInk(strokes=[stroke.smooth(window_length, polyorder) for stroke in self.strokes])
    
    def visualise(self, connect: bool=True, name: str | None = None) -> None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()

        for stroke in self.strokes:
            x = [point.x for point in stroke.points]
            y = [point.y for point in stroke.points]
            if connect:
                ax.plot(x, y, '-k', linewidth=1.5)
            else:
                ax.scatter(x, y, s=0.5, c='k')

        count = len(list(TMP_DIR.iterdir()))
        if name is None:
            name = str(count)
        else:
            name = re.sub(r'[<>:"/\\|?*]', '_', name)
            name = re.sub(r'[^\w\-_\.]', '_', name)[:50]
            name = f"{count}_{name}"
        
        # plt.show()
        # return  # for debugging

        pdf_path = TMP_DIR / f'{name}.pdf'
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close()
    
    
if __name__ == "__main__":
    digital_ink = DigitalInk.load_test()
    digital_ink.visualise()
    print(digital_ink)