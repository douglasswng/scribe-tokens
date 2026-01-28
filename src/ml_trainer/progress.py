from typing import Iterable, Sized

from tqdm import tqdm

from utils.distributed_context import distributed_context


class ProgressFactory:
    """Factory for creating progress bars with distributed awareness."""

    @staticmethod
    def create(iterable: Iterable, desc: str) -> tqdm:
        """Create a progress bar that respects distributed context."""
        total = len(iterable) if isinstance(iterable, Sized) else None
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            leave=False,
            disable=distributed_context.is_worker,
        )
