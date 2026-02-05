from constants import RESULTS_DIR
from ink_tokeniser.tokeniser import Tokeniser
from scripts.eval.utils import get_ink_iterator


RESULTS_PATH = RESULTS_DIR / "compression.csv"


def get_avg_compression_rate(tokeniser: Tokeniser) -> float:
    compression_rates = []
    for ink in get_ink_iterator():
        tokens = tokeniser.tokenise(ink)
        compression_rate = len(ink) / len(tokens)
        compression_rates.append(compression_rate)
    return sum(compression_rates) / len(compression_rates)
