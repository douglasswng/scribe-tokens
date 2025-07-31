from core.data_schema import DigitalInk
from enhancer.gen import DefaultGenerator


CANVAS_HEIGHT = 500


class Enhancer:
    def __init__(self):
        self._generator = DefaultGenerator()
        self._num_samples = 1

    def _generate_samples(self, ink: DigitalInk, text: str, strength: float) -> list[DigitalInk]:
        temperature = 1.0 - strength
        samples = self._generator.generate(ink, text, num_samples=self._num_samples, temperature=temperature)
        return samples

    def _keep_best_sample(self, samples: list[DigitalInk], text: str) -> DigitalInk:
        return samples[0]

    def _match_width(self, original_ink: DigitalInk, enhanced_ink: DigitalInk) -> DigitalInk:
        original_width = original_ink.width
        enhanced_width = enhanced_ink.width
        scale = original_width / enhanced_width
        return enhanced_ink.scale_x(scale)

    def _match_height(self, original_ink: DigitalInk, enhanced_ink: DigitalInk) -> DigitalInk:
        original_height = original_ink.height
        enhanced_height = enhanced_ink.height
        scale = original_height / enhanced_height
        return enhanced_ink.scale_y(scale)

    def _match_bbox(self, original_ink: DigitalInk, enhanced_ink: DigitalInk) -> DigitalInk:
        enhanced_ink = self._match_width(original_ink, enhanced_ink)
        enhanced_ink = self._match_height(original_ink, enhanced_ink)

        original_top_left, _ = original_ink.bbox
        current_top_left, _ = enhanced_ink.bbox
        enhanced_ink = enhanced_ink.shift(original_top_left - current_top_left)
        return enhanced_ink

    def _post_process(self, original_ink: DigitalInk, enhanced_ink: DigitalInk) -> DigitalInk:
        enhanced_ink = self._match_bbox(original_ink, enhanced_ink)
        return enhanced_ink

    def _preprocess(self, ink: DigitalInk) -> DigitalInk:
        curr_height = ink.height
        target_height = CANVAS_HEIGHT
        scale = target_height / curr_height
        return ink.scale(scale).to_origin()

    def enhance(self, ink: DigitalInk, text: str, strength: float=0.99) -> DigitalInk:
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")

        original_ink = ink.model_copy(deep=True)
        ink = self._preprocess(ink)
        samples = self._generate_samples(original_ink, text, strength)
        enhanced_ink = self._keep_best_sample(samples, text)
        return self._post_process(original_ink, enhanced_ink)


if __name__ == "__main__":
    from core.data_schema.parsed import Parsed

    parsed = Parsed.load_random()
    print(parsed.text)
    parsed.visualise()

    enhancer = Enhancer()
    ink = enhancer.enhance(parsed.ink, parsed.text)
    ink.visualise(name=parsed.text)