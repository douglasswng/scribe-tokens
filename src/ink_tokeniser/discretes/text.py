from itertools import groupby

from ink_tokeniser.discretes.discrete import DiscreteTokeniser
from ink_tokeniser.tokens import RegularToken, RegularTokenType, Token
from schemas.ink import DigitalInk, Point, Stroke
from utils.point_iterator import get_stroke_point_iterator

SEP = "␣"


class TextTokeniser(DiscreteTokeniser):
    def _create_text_token(self, char: str) -> RegularToken:
        return RegularToken(type=RegularTokenType.TEXT, values=char)

    def tokenise(self, digital_ink: DigitalInk[int]) -> list[Token]:
        tokens: list[Token] = [self.start_token]
        for point in get_stroke_point_iterator(digital_ink).rel_points:
            text = f"{point.point.x}{SEP}{point.point.y}"
            if not point.is_stroke_end:
                text += SEP
            tokens.extend([self._create_text_token(char) for char in text])
            if point.is_stroke_end:
                tokens.append(self.up_token)
        tokens.append(self.end_token)
        return tokens

    def _hybrid_str_repr(self, tokens: list[Token]) -> list[Token | str]:
        output: list[Token | str] = []
        grouped_tokens = groupby(tokens, lambda x: isinstance(x, RegularToken))
        for is_regular, group in grouped_tokens:
            if is_regular:
                regular_tokens = [token for token in group if isinstance(token, RegularToken)]
                values = [token.values for token in regular_tokens if isinstance(token.values, str)]
                text = "".join(values)
                output.append(text)
            else:
                output.extend(group)
        return output

    def _str_to_points(self, s: str) -> list[Point[int]]:  # '1␣1␣2␣-2' -> [(1, 1), (2, -2)]
        coord_strs = s.split(SEP)
        points: list[Point[int]] = []
        for i in range(0, len(coord_strs), 2):
            try:
                x = int(coord_strs[i])
                y = int(coord_strs[i + 1])
                points.append(Point(x=x, y=y))
            except (ValueError, IndexError):
                print(f"Warning: {s} cannot be converted to points, ignoring")
                continue
        return points

    def _hybrid_point_repr(self, tokens: list[Token]) -> list[Token | Point[int]]:
        hybrid_str_repr = self._hybrid_str_repr(tokens)
        output: list[Token | Point[int]] = []
        for obj in hybrid_str_repr:
            if isinstance(obj, str):
                output.extend(self._str_to_points(obj))
            else:
                output.append(obj)
        return output

    def detokenise(self, tokens: list[Token]) -> DigitalInk[int]:
        current_point: Point[int] = Point(x=0, y=0)
        stroke: Stroke[int] = Stroke(points=[current_point])
        strokes: list[Stroke[int]] = []
        for obj in self._hybrid_point_repr(tokens):
            match obj:
                case self.start_token:
                    continue
                case self.end_token:
                    return DigitalInk(strokes=strokes)
                case self.up_token:
                    strokes.append(stroke)
                    stroke = Stroke(points=[])
                case Point():
                    current_point += obj
                    stroke.points.append(current_point)
                case _:
                    raise ValueError(f"Unknown token: {obj}")
        return DigitalInk(strokes=strokes)


if __name__ == "__main__":
    from schemas.parsed import Parsed

    parsed = Parsed.load_random()
    digital_ink = parsed.ink
    digital_ink.visualise()

    tokeniser = TextTokeniser()
    tokens = tokeniser.tokenise(digital_ink)
    print("\n".join([str(token) for token in tokens]))

    detokenised = tokeniser.detokenise(tokens)
    detokenised.visualise()
