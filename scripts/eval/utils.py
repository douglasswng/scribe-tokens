from typing import Iterator

from dataloader.split import create_datasplit
from ink_tokeniser.tokeniser import Tokeniser
from schemas.ink import DigitalInk
from schemas.parsed import Parsed


def get_ink_iterator() -> Iterator[DigitalInk]:
    datasplit = create_datasplit()
    for path in datasplit.val_paths[:]:
        parsed = Parsed.from_path(path)
        yield parsed.ink.to_origin()


def get_tokeniser_iterator() -> Iterator[Tokeniser]: ...
