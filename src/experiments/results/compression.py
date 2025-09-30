from ..utils import get_ink_iterator, get_tokeniser_iterator, add_results, finalise_results
from ...schemas import TokenReprType, TokeniserFramework
from ...constants import RESULTS_DIR

RESULT_PATH = RESULTS_DIR / 'compression.csv'
RESULT_NAME = 'compression_rate'
TOKENISER_TYPES = [TokenReprType.SCRIBE_TOKENS, TokenReprType.REL_TOKENS, TokenReprType.ABS_TOKENS, TokenReprType.TEXT_TOKENS]
DELTAS = [1, 2, 4, 8, 16, 32]
VOCAB_SIZES = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]

def get_precalculated_compression_rate(tokeniser: TokeniserFramework) -> float:
    if tokeniser.tokeniser_id.type == TokenReprType.REL_TOKENS:
        return 0.9590406277513334
    elif tokeniser.tokeniser_id.type == TokenReprType.ABS_TOKENS:
        return 0.9573421912405401
    else:
        raise ValueError(f"Tokeniser of type {tokeniser.tokeniser_id.type} should have merges")

def get_avg_compression_rate(tokeniser: TokeniserFramework) -> float:
    if not tokeniser.has_fast:
        return get_precalculated_compression_rate(tokeniser)
        
    compression_rates = []
    for ink in get_ink_iterator():
        if not tokeniser.has_fast:
            print(f"Tokeniser {tokeniser.tokeniser_id!s} has no fast tokeniser, "
                  f"using base tokeniser with no merges. "
                  f"This can be precalculated to speed up experiment")
            tokens = tokeniser.base_tokenise(ink)
        else:
            tokens = tokeniser.tokenise(ink)
        compression_rate = len(ink) / len(tokens)
        compression_rates.append(compression_rate)
    return sum(compression_rates) / len(compression_rates)

def compression() -> None:
    for tokeniser in get_tokeniser_iterator(RESULT_PATH, TOKENISER_TYPES, DELTAS, VOCAB_SIZES):
        result = get_avg_compression_rate(tokeniser)
        print(f"Compression rate for {tokeniser.tokeniser_id!s} (vocab={tokeniser.vocab_size}): {result}")
        add_results(RESULT_PATH, RESULT_NAME, tokeniser, result)
    finalise_results(RESULT_PATH)

if __name__ == '__main__':
    compression()