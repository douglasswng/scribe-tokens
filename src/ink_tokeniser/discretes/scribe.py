from typing import Literal

from ink_tokeniser.discretes.discrete import DiscreteTokeniser
from ink_tokeniser.tokens import RegularToken, RegularTokenType, Token
from schemas.ink import DigitalInk, Point, Stroke
from utils.point_iterator import get_stroke_point_iterator

COORD_TO_STR: dict[tuple[int, int], str] = {
    (0, 1): "↑",
    (0, -1): "↓",
    (-1, 0): "←",
    (1, 0): "→",
    (-1, 1): "↖",
    (1, 1): "↗",
    (-1, -1): "↙",
    (1, -1): "↘",
}

STR_TO_COORD: dict[str, tuple[int, int]] = {v: k for k, v in COORD_TO_STR.items()}


def bres_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    points = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


class ScribeTokeniser(DiscreteTokeniser):
    def _create_scribe_token(self, text: str) -> RegularToken:
        return RegularToken(type=RegularTokenType.SCRIBE, values=text)

    def _point_to_token(self, point: Point[int]) -> RegularToken:
        return self._create_scribe_token(COORD_TO_STR[(point.x, point.y)])

    def _token_to_point(self, token: RegularToken) -> Point[int]:
        assert isinstance(token.values, str)
        coord = STR_TO_COORD[token.values]
        return Point(x=coord[0], y=coord[1])

    def _bres_line_decomp(self, delta: Point[int]) -> list[RegularToken]:
        pixels = bres_line(0, 0, delta.x, delta.y)
        pixel_points = [Point(x=pixel[0], y=pixel[1]) for pixel in pixels]
        rel_points = [p2 - p1 for p1, p2 in zip(pixel_points, pixel_points[1:])]
        tokens: list[RegularToken] = []
        for rel_point in rel_points:
            tokens.append(self._point_to_token(rel_point))
        return tokens

    def tokenise(self, digital_ink: DigitalInk[int]) -> list[Token]:
        tokens: list[Token] = [self.start_token, self.down_token]
        for point in get_stroke_point_iterator(digital_ink).rel_points:
            tokens.extend([scribe_token for scribe_token in self._bres_line_decomp(point.point)])
            if point.is_stroke_start:
                tokens.append(self.down_token)
            if point.is_stroke_end:
                tokens.append(self.up_token)
        tokens.append(self.end_token)
        return tokens

    def detokenise(self, tokens: list[Token]) -> DigitalInk[int]:
        current_point: Point[int] = Point(x=0, y=0)
        strokes: list[Stroke[int]] = [Stroke(points=[current_point])]
        pen_state: Literal["up", "down"] = "down"
        for token in tokens:
            match token, pen_state:
                case self.start_token, _:
                    continue
                case self.end_token, _:
                    return DigitalInk(strokes=strokes)
                case self.up_token, "up":
                    continue
                case self.down_token, "down":
                    continue
                case self.up_token, "down":
                    pen_state = "up"
                case self.down_token, "up":
                    pen_state = "down"
                    strokes.append(Stroke(points=[current_point]))
                case RegularToken(), "up":
                    current_point += self._token_to_point(token)
                case RegularToken(), "down":
                    current_point += self._token_to_point(token)
                    strokes[-1].points.append(current_point)
                case _:
                    raise ValueError(f"Unknown token: {token}")
        return DigitalInk(strokes=strokes)


if __name__ == "__main__":
    from schemas.parsed import Parsed

    parsed = Parsed.load_random()
    digital_ink = parsed.ink
    digital_ink.visualise()

    tokeniser = ScribeTokeniser()
    tokens = tokeniser.tokenise(digital_ink)
    print("\n".join([str(token) for token in tokens]))

    detokenised = tokeniser.detokenise(tokens)
    detokenised.visualise()
