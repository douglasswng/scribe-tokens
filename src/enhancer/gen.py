from typing import Protocol

from core.data_schema import DigitalInk, InstancePair, Instance, Parsed
from core.model import ModelId, Task
from core.repr import TokenReprId
from model.factory import DefaultModelFactory
from model.models.generation import GenerationModel
from enhancer.utils import InkTextConverter


class Generator(Protocol):
    _model: GenerationModel

    def _load_instance_by_ink(self, ink: DigitalInk) -> Instance:
        parsed = Parsed(id='', text='', writer='', ink=ink)
        repr = InkTextConverter.ink_to_tensor(ink)
        return Instance(parsed=parsed, _repr=repr)

    def _load_instance_by_text(self, text: str) -> Instance:
        parsed = Parsed(id='', text=text, writer='', ink=DigitalInk(strokes=[]))
        repr = InkTextConverter.text_to_tensor(text)
        return Instance(parsed=parsed, _repr=repr)

    def _load_instance_pair(self, ref_ink: DigitalInk, text: str) -> InstancePair:
        ref_instance = self._load_instance_by_ink(ref_ink)
        main_instance = self._load_instance_by_text(text)
        return InstancePair(main_instance=main_instance,
                            ref_instance=ref_instance)

    def generate(self, ref_ink: DigitalInk, text: str,
                 num_samples: int=1,
                 temperature: float=0.3) -> list[DigitalInk]:
        instance_pair = self._load_instance_pair(ref_ink, text)
        gen_inks = self._model.generate_inks(instance_pair, num_gen=num_samples, temperature=temperature)
        return gen_inks


class DefaultGenerator(Generator):
    def __init__(self):
        self._model = self._load_model()

    def _load_model(self) -> GenerationModel:
        model_id = ModelId(task=Task.GENERATION, repr_id=TokenReprId.create_scribe())
        model = DefaultModelFactory.load_pretrained(model_id)
        assert isinstance(model, GenerationModel)
        assert not model.training
        return model


if __name__ == "__main__":
    from core.data_schema.parsed import Parsed

    parsed = Parsed.load_random()
    print(parsed.text)
    parsed.visualise()

    gen = DefaultGenerator()

    ink = gen.generate(parsed.ink, parsed.text)[0]
    ink.visualise(name=parsed.text)