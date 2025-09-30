from pathlib import Path
from typing import Iterator
from itertools import product
import csv
from ..tokeniser import TokeniserFactory
from ..data import DataSplit
from ..schemas import TokenReprId, TokenReprType, TokeniserFramework, DigitalInk

PERSISTENT_COLUMNS = ('tokeniser_type', 'delta', 'vocab_size')

def get_ink_iterator(data_split: DataSplit = DataSplit()) -> Iterator[DigitalInk]:
    for iam in data_split.val_iams:
        yield DigitalInk.from_iam(iam)

def seen_tokeniser(result_path: Path, tokeniser_id: TokenReprId, vocab_size: int) -> bool:
    with open(result_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if row[0] == tokeniser_id.type.value and \
                row[1] == str(tokeniser_id.delta) and \
                row[2] == str(vocab_size):
                 return True
    return False

def get_tokeniser_iterator(result_path: Path,
                           types: list[TokenReprType],
                           deltas: list[int],
                           vocab_sizes: list[int]) -> Iterator[TokeniserFramework]:
    for type, delta, vocab_size in product(types, deltas, vocab_sizes):
        tokeniser_id = TokenReprId(delta=delta, type=type)
        if seen_tokeniser(result_path, tokeniser_id, vocab_size):
            print(f"Skipping {tokeniser_id!s} (vocab={vocab_size}) because it has already been seen")
            continue
        tokeniser = TokeniserFactory.create_tokeniser(tokeniser_id, vocab_size=vocab_size)
        yield tokeniser

def add_results(result_path: Path,
                result_name: str,  # e.g. 'compression rate'
                tokeniser: TokeniserFramework,
                result: float,
                persistent_columns: tuple[str, ...]=PERSISTENT_COLUMNS) -> None:
    columns = list(persistent_columns) + [result_name]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if not result_path.exists() or result_path.stat().st_size == 0:
        with open(result_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    tokeniser_type = tokeniser.tokeniser_id.type.value
    delta = tokeniser.tokeniser_id.delta
    vocab_size = tokeniser.vocab_size
    with open(result_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([tokeniser_type, delta, vocab_size, result])

def finalise_results(result_path: Path) -> None:
    with open(result_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return
        rows = list(reader)
    
    rows.sort(key=lambda row: (row[0], int(row[1]), int(row[2])))
    
    with open(result_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)