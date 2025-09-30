from pathlib import Path
from typing import Iterator, Callable, Iterable
from itertools import product
import csv
from multiprocessing import Pool
from dataloader.split import DataSplit, create_datasplit
from core.repr.id import TokenReprId, TokenReprType
from core.data_schema import DigitalInk, Parsed

PERSISTENT_COLUMNS = ('tokeniser_type', 'delta', 'vocab_size')


def get_ink_iterator(data_split: DataSplit = create_datasplit()) -> Iterator[DigitalInk]:
    for path in data_split.val_paths[:]:
        parsed = Parsed.from_path(path)
        yield parsed.ink.to_origin()


def seen_tokeniser(result_path: Path, tokeniser_id: TokenReprId) -> bool:
    if not result_path.exists():
        return False

    with open(result_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if row[0] == tokeniser_id.type.value and \
                row[1] == str(tokeniser_id.delta) and \
                row[2] == str(tokeniser_id.vocab_size):
                 return True
    return False


def get_tokeniser_id_iterator(result_path: Path,
                              types: list[TokenReprType],
                              deltas: list[int],
                              vocab_sizes: list[int]) -> Iterator[TokenReprId]:
    for type, delta, vocab_size in product(types, deltas, vocab_sizes):
        tokeniser_id = TokenReprId(delta=delta, type=type, vocab_size=vocab_size)
        if not tokeniser_id.has_trained:
            print(f"Skipping {tokeniser_id!s} (vocab={vocab_size}) because it does not exist")
            continue
        if seen_tokeniser(result_path, tokeniser_id):
            print(f"Skipping {tokeniser_id!s} (vocab={vocab_size}) because it has already been seen")
            continue
        yield tokeniser_id


def add_results(result_path: Path,
                result_name: str,  # e.g. 'compression rate'
                tokeniser_id: TokenReprId,
                result: float,
                persistent_columns: tuple[str, ...]=PERSISTENT_COLUMNS) -> None:
    columns = list(persistent_columns) + [result_name]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if not result_path.exists() or result_path.stat().st_size == 0:
        with open(result_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    tokeniser_type = tokeniser_id.type.value
    delta = tokeniser_id.delta
    vocab_size = tokeniser_id.vocab_size
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


def run_experiment_multiprocess(
    tokeniser_ids: Iterable[TokenReprId],
    experiment_func: Callable[[TokenReprId], tuple[TokenReprId, float] | None],
    result_path: Path,
    result_name: str,
    max_workers: int | None = None
) -> None:
    """
    Run an experiment function on multiple tokeniser IDs using multiprocessing.
    
    Args:
        tokeniser_ids: Iterable of TokenReprId to process
        experiment_func: Function that takes a TokenReprId and returns (TokenReprId, result) or None to skip
        result_path: Path to save results CSV
        result_name: Name of the result column (e.g., 'compression_rate')
        max_workers: Maximum number of processes to use. If None, defaults to the number of processors.
    """
    tokeniser_id_list = list(tokeniser_ids)
    
    if not tokeniser_id_list:
        print("No tokenisers to process")
        finalise_results(result_path)
        return
    
    print(f"Processing {len(tokeniser_id_list)} tokenisers with multiprocessing...")
    
    # Use multiprocessing Pool with process-safe result writing
    with Pool(processes=max_workers) as pool:
        # Map the experiment function to all tokeniser IDs
        results = pool.map(experiment_func, tokeniser_id_list)
    
    # Write all results (sequentially after processing is done), filtering out None values
    for result in results:
        if result is not None:
            tokeniser_id, value = result
            add_results(result_path, result_name, tokeniser_id, value)
    
    finalise_results(result_path)