from typing import Iterator

from constants import DATASET
from dataloader.split import DataSplit, create_datasplit
from ink_tokeniser.factory import TokeniserId, TokenType
from ink_tokeniser.trainer import InkBpeTrainer
from schemas.ink import DigitalInk
from schemas.parsed import Parsed

assert DATASET == "iam", "Use IAM for more robust tokenisers"

DELTAS = [1, 2, 4, 8, 16, 32]
MAX_VOCAB_SIZE = 100_000


def get_id_iterator() -> Iterator[TokeniserId]:
    for delta in reversed(DELTAS):  # larger deltas train quicker
        for type in TokenType:
            yield TokeniserId(type=type, delta=delta, vocab_size=MAX_VOCAB_SIZE)


def get_ink_iterator(data_split: DataSplit) -> Iterator[DigitalInk]:
    for path in data_split.train_paths[:]:
        parsed = Parsed.from_path(path)
        yield parsed.ink.to_origin()


def train_tokenisers() -> None:
    data_split = create_datasplit()
    for id in get_id_iterator():
        if id.tokeniser_path.exists():
            print(f"Tokeniser {id} already exists, skipping...")
            continue
        trainer = InkBpeTrainer(id, MAX_VOCAB_SIZE)
        ink_iterator = get_ink_iterator(data_split)
        try:
            trainer.train_from_iterator(ink_iterator)
            print(f"Trained tokeniser: {id}")
        except BaseException as e:  # run out of unicode
            print(e)
            print(f"{id!s} is merge-ineligible, skipping...")


if __name__ == "__main__":
    train_tokenisers()
