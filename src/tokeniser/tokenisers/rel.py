from core.data_schema import DigitalInk, Point, Stroke
from core.tokeniser import RegularToken, RegularTokenType, Token, DiscreteTokeniser, Coord
from core.utils import get_stroke_point_iterator


class RelTokeniser(DiscreteTokeniser):
    def _create_rel_token(self, dx: int, dy: int) -> RegularToken:
        return RegularToken(type=RegularTokenType.REL,
                            values=[Coord(x=dx, y=dy)])
    
    def _create_point(self, token: RegularToken) -> Point:
        coord = token.values[0]
        assert isinstance(coord, Coord)
        return Point(x=coord.x, y=coord.y)

    def tokenise(self, digital_ink: DigitalInk[int]) -> list[Token]:
        tokens: list[Token] = [self.start_token]
        for point in get_stroke_point_iterator(digital_ink).rel_points:
            rel_token = self._create_rel_token(point.x, point.y)
            tokens.append(rel_token)
            if point.is_stroke_end:
                tokens.append(self.up_token)
        tokens.append(self.end_token)
        return tokens

    def detokenise(self, tokens: list[Token]) -> DigitalInk[int]:
        current_point: Point[int] = Point(x=0, y=0)
        stroke: Stroke[int] = Stroke(points=[current_point])
        strokes: list[Stroke[int]] = []
        for token in tokens:
            match token:
                case self.start_token:
                    continue
                case self.unknown_token:
                    continue
                case self.end_token:
                    return DigitalInk(strokes=strokes)
                case self.up_token:
                    strokes.append(stroke)
                    stroke = Stroke(points=[])
                case RegularToken():
                    point = self._create_point(token)
                    current_point = current_point + point
                    stroke.points.append(current_point)
                case _:
                    raise ValueError(f"Unknown token: {token}")
        return DigitalInk(strokes=strokes)


if __name__ == "__main__":
    from core.data_schema import Parsed
    parsed = Parsed.load_random()
    digital_ink = parsed.ink
    digital_ink.visualise()

    tokeniser = RelTokeniser()
    tokens = tokeniser.tokenise(digital_ink)
    print('\n'.join([str(token) for token in tokens]))
    
    detokenised = tokeniser.detokenise(tokens)
    detokenised.visualise()