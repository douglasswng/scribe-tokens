from core.tokeniser.tokens import Token
from core.tokeniser.preprocessor import Preprocessor
from core.tokeniser.discrete import DiscreteTokeniser
from core.tokeniser.trained import TrainedTokeniser
from core.data_schema import DigitalInk


class Tokeniser:
    def __init__(self,
                 preprocessor: Preprocessor,
                 discrete_tokeniser: DiscreteTokeniser,
                 trained_tokeniser: TrainedTokeniser | None = None):
        self._preprocessor = preprocessor
        self._discrete_tokeniser = discrete_tokeniser
        self._trained_tokeniser = trained_tokeniser

    @property
    def unk_token_id(self) -> int | None:
        if self._trained_tokeniser is None:
            raise ValueError("Trained tokeniser is not set")
        return self._trained_tokeniser.unk_token_id

    def tokenise(self, digital_ink: DigitalInk) -> list[Token]:
        tokens = self._preprocessor.preprocess(digital_ink)
        tokens = self._discrete_tokeniser.tokenise(tokens)
        if self._trained_tokeniser is not None:
            tokens = self._trained_tokeniser.merge(tokens)
        return tokens
    
    def detokenise(self, tokens: list[Token]) -> DigitalInk:
        if self._trained_tokeniser is not None:
            tokens = self._trained_tokeniser.split(tokens)
        digital_ink = self._discrete_tokeniser.detokenise(tokens)
        digital_ink = self._preprocessor.postprocess(digital_ink)
        return digital_ink
    
    def convert_tokens_to_ids(self, tokens: list[Token]) -> list[int]:
        if self._trained_tokeniser is None:
            raise ValueError("Trained tokeniser is not set")
        return self._trained_tokeniser.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids: list[int]) -> list[Token]:
        if self._trained_tokeniser is None:
            raise ValueError("Trained tokeniser is not set")
        return self._trained_tokeniser.convert_ids_to_tokens(ids)

    def encode(self, digital_ink: DigitalInk) -> list[int]:
        tokens = self.tokenise(digital_ink)
        ids = self.convert_tokens_to_ids(tokens)
        return ids
    
    def decode(self, ids: list[int]) -> DigitalInk:
        tokens = self.convert_ids_to_tokens(ids)
        digital_ink = self.detokenise(tokens)
        return digital_ink