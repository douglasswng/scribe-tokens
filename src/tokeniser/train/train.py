from typing import Iterator
from dataloader.split import DataSplit, create_datasplit
from tokeniser.train.trainer import InkBpeTrainer
from core.repr.id import TokenReprId, TokenReprType
from core.data_schema import DigitalInk, Parsed


DELTAS = [1, 2, 4, 8, 16, 32]
MAX_VOCAB_SIZE = 100_000


def get_id_iterator() -> Iterator[TokenReprId]:
    for delta in reversed(DELTAS):  # larger deltas train quicker
        for type in TokenReprType:
            yield TokenReprId(delta=delta, type=type)


def get_ink_iterator(data_split: DataSplit) -> Iterator[DigitalInk]:
    for path in data_split.train_paths:
        parsed = Parsed.from_path(path)
        yield parsed.ink


def train_tokenisers() -> None:
    data_split = create_datasplit()
    for id in get_id_iterator():
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