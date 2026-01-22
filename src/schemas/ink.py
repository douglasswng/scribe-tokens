import io
from typing import overload

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
from pydantic import BaseModel
from scipy.signal import savgol_filter

from constants import TMP_DIR


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

    def __add__(self, other: "Point[T]") -> "Point[T]":
        return Point(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: "Point[T]") -> "Point[T]":
        return Point(x=self.x - other.x, y=self.y - other.y)

    def __neg__(self) -> "Point[T]":
        return Point(x=-self.x, y=-self.y)

    def __mul__(self, other: float) -> "Point":
        return Point(x=self.x * other, y=self.y * other)

    def round(self) -> "Point[int]":
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

    def shift(self, shift: Point) -> "Stroke":
        return Stroke(points=[point + shift for point in self.points])

    def scale(self, scale: float) -> "Stroke":
        return Stroke(points=[point * scale for point in self.points])

    def discretise(self) -> "Stroke[int]":
        return Stroke(points=[point.round() for point in self.points])

    def downsample(self, factor: float) -> "Stroke":
        if len(self.points) <= 2:
            return Stroke(points=self.points[:])

        downsampled = [self.points[0]]

        i = factor
        while i < len(self.points) - 1:
            downsampled.append(self.points[int(i)])
            i += factor

        downsampled.append(self.points[-1])

        return Stroke(points=downsampled)

    def smooth(self, window_length: int, polyorder: int) -> "Stroke":
        if len(self.points) < window_length:
            return Stroke(points=self.points[:])

        x_coords = np.array([point.x for point in self.points])
        y_coords = np.array([point.y for point in self.points])

        smoothed_x = savgol_filter(x_coords, window_length, polyorder)
        smoothed_y = savgol_filter(y_coords, window_length, polyorder)
        
        assert isinstance(smoothed_x, np.ndarray)
        assert isinstance(smoothed_y, np.ndarray)

        smoothed_points = [Point(x=float(x), y=float(y)) for x, y in zip(smoothed_x, smoothed_y)]
        return Stroke(points=smoothed_points)

    def flip_y(self) -> "Stroke":
        return Stroke(points=[Point(x=point.x, y=-point.y) for point in self.points])

    def scale_x(self, scale: float) -> "Stroke":
        return Stroke(points=[Point(x=point.x * scale, y=point.y) for point in self.points])

    def scale_y(self, scale: float) -> "Stroke":
        return Stroke(points=[Point(x=point.x, y=point.y * scale) for point in self.points])

    def dedup(self) -> "Stroke":
        deduped_points = [self.points[0]]
        for point in self.points[1:]:
            if point != deduped_points[-1]:
                deduped_points.append(point)
        return Stroke(points=deduped_points)


class DigitalInk[T: (float, int)](BaseModel):
    strokes: list[Stroke[T]]

    def __reduce_ex__(self, protocol):
        return (_reconstruct_digital_ink, (self.strokes,))

    def __str__(self) -> str:
        line = "-" * 100
        strokes_str = f"{line}\nDigitalInk:\n"
        strokes_str += "\n\n".join(
            f"  stroke{i + 1}: {stroke}" for i, stroke in enumerate(self.strokes)
        )
        strokes_str += f"\n{line}"
        return strokes_str

    def __len__(self) -> int:
        return sum(len(stroke) for stroke in self.strokes)

    @overload
    @classmethod
    def from_coords(cls, raw_strokes: list[list[tuple[int, int]]]) -> "DigitalInk[int]": ...

    @overload
    @classmethod
    def from_coords(cls, raw_strokes: list[list[tuple[float, float]]]) -> "DigitalInk[float]": ...

    @overload
    @classmethod
    def from_coords(cls, raw_strokes: list[np.ndarray]) -> "DigitalInk": ...

    @classmethod
    def from_coords(cls, raw_strokes):
        strokes = []
        for stroke in raw_strokes:
            points = [Point(x=coord[0], y=coord[1]) for coord in stroke]
            strokes.append(Stroke(points=points))
        digital_ink = DigitalInk(strokes=strokes)
        return digital_ink

    @classmethod
    def load_test(cls) -> "DigitalInk[int]":
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

    def to_coords(self) -> list[list[tuple[float, float]]]:
        return [[(point.x, point.y) for point in stroke.points] for stroke in self.strokes]

    def to_image(self) -> Image.Image:
        fig, ax = self._create_plot()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        return img

    def to_origin(self) -> "DigitalInk":
        start = self.strokes[0].points[0]
        return self.shift(-start)

    def shift(self, shift: Point) -> "DigitalInk":
        return DigitalInk(strokes=[stroke.shift(shift) for stroke in self.strokes])

    def scale(self, scale: float) -> "DigitalInk":
        return DigitalInk(strokes=[stroke.scale(scale) for stroke in self.strokes])

    def scale_x(self, scale: float) -> "DigitalInk":
        return DigitalInk(strokes=[stroke.scale_x(scale) for stroke in self.strokes])

    def scale_y(self, scale: float) -> "DigitalInk":
        return DigitalInk(strokes=[stroke.scale_y(scale) for stroke in self.strokes])

    def discretise(self) -> "DigitalInk[int]":
        return DigitalInk(strokes=[stroke.discretise() for stroke in self.strokes])

    def downsample(self, factor: float) -> "DigitalInk":
        return DigitalInk(strokes=[stroke.downsample(factor) for stroke in self.strokes])

    def smooth(self, window_length: int, polyorder: int) -> "DigitalInk":
        return DigitalInk(
            strokes=[stroke.smooth(window_length, polyorder) for stroke in self.strokes]
        )

    def flip_y(self) -> "DigitalInk":
        return DigitalInk(strokes=[stroke.flip_y() for stroke in self.strokes])

    def dedup(self) -> "DigitalInk":
        return DigitalInk(strokes=[stroke.dedup() for stroke in self.strokes])

    def visualise(self, connect: bool = True, name: str | None = None) -> None:
        TMP_DIR.mkdir(parents=True, exist_ok=True)

        fig, ax = self._create_plot(connect=connect)

        count = len(list(TMP_DIR.iterdir()))
        name = str(count) if name is None else f"{count}_{name}"
        name = name[:100]

        pdf_path = TMP_DIR / f"{name}.pdf"
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

    def _create_plot(
        self, connect: bool = True, figsize: tuple[int, int] = (12, 8)
    ) -> tuple[Figure, Axes]:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()

        for stroke in self.strokes:
            x = [point.x for point in stroke.points]
            y = [point.y for point in stroke.points]

            if len(stroke.points) == 1:
                ax.scatter(x, y, s=10, c="k")
            elif connect:
                ax.plot(x, y, "-k", linewidth=1.5)
            else:
                ax.scatter(x, y, s=0.5, c="k")

        return fig, ax


if __name__ == "__main__":
    digital_ink = DigitalInk.load_test()
    digital_ink.visualise()
    print(digital_ink)
