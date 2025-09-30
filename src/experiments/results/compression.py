from typing import Iterable
from experiments.utils import get_ink_iterator, run_experiment_multiprocess
from core.repr.id import TokenReprType, TokenReprId
from core.tokeniser import Tokeniser
from core.constants import RESULTS_DIR
from tokeniser.factory import DefaultTokeniserFactory

RESULT_PATH = RESULTS_DIR / 'compression.csv'
RESULT_NAME = 'compression_rate'


def get_avg_compression_rate(tokeniser: Tokeniser) -> float:
    compression_rates = []
    for ink in get_ink_iterator():
        tokens = tokeniser.tokenise(ink)
        compression_rate = len(ink) / len(tokens)
        compression_rates.append(compression_rate)
    return sum(compression_rates) / len(compression_rates)


def process_tokeniser(tokeniser_id: TokenReprId) -> tuple[TokenReprId, float]:
    """Process a single tokeniser and return its ID and compression rate."""
    tokeniser = DefaultTokeniserFactory.create(tokeniser_id)
    result = get_avg_compression_rate(tokeniser)
    print(f"Compression rate for {tokeniser_id!s} (vocab={tokeniser_id.vocab_size}): {result}")
    return tokeniser_id, result


def compression(tokeniser_ids: Iterable[TokenReprId], max_workers: int | None = None) -> None:
    run_experiment_multiprocess(
        tokeniser_ids=tokeniser_ids,
        experiment_func=process_tokeniser,
        result_path=RESULT_PATH,
        result_name=RESULT_NAME,
        max_workers=max_workers
    )


if __name__ == '__main__':
    from experiments.utils import get_tokeniser_id_iterator
    
    TOKENISER_TYPES = [TokenReprType.SCRIBE, TokenReprType.REL, TokenReprType.ABS, TokenReprType.TEXT]
    DELTAS = [1, 2, 4, 8, 16, 32]
    VOCAB_SIZES = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]
    
    tokeniser_ids = get_tokeniser_id_iterator(RESULT_PATH, TOKENISER_TYPES, DELTAS, VOCAB_SIZES)
    
    compression(tokeniser_ids, max_workers=192)