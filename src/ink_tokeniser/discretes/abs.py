from ink_tokeniser.discretes.discrete import DiscreteTokeniser
from ink_tokeniser.tokens import Coord, RegularToken, RegularTokenType, Token
from schemas.ink import DigitalInk, Point, Stroke


class AbsTokeniser(DiscreteTokeniser):
    def _create_abs_token(self, x: int, y: int) -> RegularToken:
        return RegularToken(type=RegularTokenType.ABS, values=[Coord(x=x, y=y)])

    def _create_point(self, token: RegularToken) -> Point:
        coord = token.values[0]
        assert isinstance(coord, Coord)
        return Point(x=coord.x, y=coord.y)

    def tokenise(self, ink: DigitalInk[int]) -> list[Token]:
        tokens: list[Token] = [self.start_token]
        for stroke in ink.strokes:
            for point in stroke.points:
                tokens.append(self._create_abs_token(point.x, point.y))
            tokens.append(self.up_token)
        tokens.append(self.end_token)
        return tokens

    def detokenise(self, tokens: list[Token]) -> DigitalInk[int]:
        stroke: Stroke[int] = Stroke(points=[])
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
                    stroke.points.append(point)
                case _:
                    raise ValueError(f"Unknown token: {token}")
        return DigitalInk(strokes=strokes)
