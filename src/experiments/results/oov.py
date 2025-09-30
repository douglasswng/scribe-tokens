from ..utils import get_ink_iterator, get_tokeniser_iterator, add_results, finalise_results
from ...schemas import TokenReprType, TokeniserFramework, SpecialTokenType
from ...constants import RESULTS_DIR

RESULT_PATH = RESULTS_DIR / 'oov.csv'
RESULT_NAME = 'oov_rate'
TOKENISER_TYPES = [TokenReprType.REL_TOKENS, TokenReprType.ABS_TOKENS]
DELTAS = [1, 2, 4, 8, 16, 32]
VOCAB_SIZES = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]

def get_avg_oov_rate(tokeniser: TokeniserFramework) -> float:
    oov_rates = []
    for ink in get_ink_iterator():
        token_ids = tokeniser.encode(ink)
        tokens = tokeniser.convert_ids_to_tokens(token_ids)
        unknown_tokens = [token for token in tokens
                          if token.type == SpecialTokenType.UNKNOWN]
        oov_rate = len(unknown_tokens) / len(tokens)
        oov_rates.append(oov_rate)
    return sum(oov_rates) / len(oov_rates)

def oov() -> None:
    for tokeniser in get_tokeniser_iterator(RESULT_PATH, TOKENISER_TYPES, DELTAS, VOCAB_SIZES):
        if not tokeniser.has_fast:
            print(f"Skipping tokeniser {tokeniser.tokeniser_id!s} (vocab={tokeniser.vocab_size})"
                  f"because it is merge-ineligible")
            continue
        result = get_avg_oov_rate(tokeniser)
        add_results(RESULT_PATH, RESULT_NAME, tokeniser, result)
    finalise_results(RESULT_PATH)

if __name__ == '__main__':
    oov()