from experiments.utils import get_ink_iterator, get_tokeniser_iterator, add_results, finalise_results
from core.repr.id import TokenReprType, TokenReprId
from core.tokeniser import Tokeniser
from core.constants import RESULTS_DIR
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

RESULT_PATH = RESULTS_DIR / 'compression.csv'
RESULT_NAME = 'compression_rate'
TOKENISER_TYPES = [TokenReprType.SCRIBE, TokenReprType.REL, TokenReprType.ABS, TokenReprType.TEXT]
DELTAS = [1, 2, 4, 8, 16, 32]
VOCAB_SIZES = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]

# Thread-safe lock for writing results
write_lock = Lock()


def get_avg_compression_rate(tokeniser: Tokeniser) -> float:
    compression_rates = []
    for ink in get_ink_iterator():
        tokens = tokeniser.tokenise(ink)
        compression_rate = len(ink) / len(tokens)
        compression_rates.append(compression_rate)
    return sum(compression_rates) / len(compression_rates)


def process_tokeniser(tokeniser_id: TokenReprId, tokeniser: Tokeniser) -> tuple[TokenReprId, float]:
    """Process a single tokeniser and return its ID and compression rate."""
    result = get_avg_compression_rate(tokeniser)
    print(f"Compression rate for {tokeniser_id!s} (vocab={tokeniser_id.vocab_size}): {result}")
    return tokeniser_id, result


def load_tokeniser(tokeniser_id: TokenReprId) -> tuple[TokenReprId, Tokeniser | None]:
    """Load a single tokeniser and return its ID and instance."""
    from tokeniser.factory import DefaultTokeniserFactory
    try:
        tokeniser = DefaultTokeniserFactory.create(tokeniser_id)
        return tokeniser_id, tokeniser
    except Exception as exc:
        print(f"Failed to load tokeniser {tokeniser_id!s}: {exc}")
        return tokeniser_id, None


def compression(max_workers: int | None = None) -> None:
    """
    Run compression rate experiments with multithreading.
    
    Args:
        max_workers: Maximum number of threads to use. If None, defaults to the number of processors.
    """
    # Collect all tokeniser IDs first (without loading them yet)
    tokeniser_ids = []
    for tokeniser_id, _ in get_tokeniser_iterator(RESULT_PATH, TOKENISER_TYPES, DELTAS, VOCAB_SIZES):
        tokeniser_ids.append(tokeniser_id)
    
    if not tokeniser_ids:
        print("No tokenisers to process")
        finalise_results(RESULT_PATH)
        return
    
    print(f"Loading {len(tokeniser_ids)} tokenisers in parallel...")
    
    # Load all tokenisers in parallel
    tokeniser_tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(load_tokeniser, tokeniser_id): tokeniser_id
            for tokeniser_id in tokeniser_ids
        }
        
        for future in as_completed(future_to_id):
            tokeniser_id, tokeniser = future.result()
            if tokeniser is not None:
                tokeniser_tasks.append((tokeniser_id, tokeniser))
    
    if not tokeniser_tasks:
        print("No tokenisers successfully loaded")
        finalise_results(RESULT_PATH)
        return
    
    print(f"Processing {len(tokeniser_tasks)} tokenisers with multithreading...")
    
    # Process tokenisers in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_tokeniser = {
            executor.submit(process_tokeniser, tokeniser_id, tokeniser): tokeniser_id
            for tokeniser_id, tokeniser in tokeniser_tasks
        }
        
        # Process results as they complete
        for future in as_completed(future_to_tokeniser):
            tokeniser_id = future_to_tokeniser[future]
            try:
                tokeniser_id, result = future.result()
                # Write results in a thread-safe manner
                with write_lock:
                    add_results(RESULT_PATH, RESULT_NAME, tokeniser_id, result)
            except Exception as exc:
                print(f"Tokeniser {tokeniser_id!s} generated an exception: {exc}")
    
    finalise_results(RESULT_PATH)


if __name__ == '__main__':
    compression(max_workers=192)