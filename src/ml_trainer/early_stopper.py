from ml_trainer.stats import EpochStats


class EarlyStopper:
    def __init__(self, patience: int):
        self._patience = patience
        self._best_loss = float("inf")
        self._counter = 0
        self._improved = False

    @property
    def counter(self) -> int:
        return self._counter

    @property
    def patience(self) -> int:
        return self._patience

    @property
    def is_best(self) -> bool:
        return self._improved

    @property
    def should_stop(self) -> bool:
        return self._counter >= self._patience

    def register_stats(self, val_epoch_stats: EpochStats) -> None:
        """Register stats for early stopping."""
        self._improved = False

        if val_epoch_stats.loss < self._best_loss:
            self._best_loss = val_epoch_stats.loss
            self._counter = 0
            self._improved = True
        else:
            self._counter += 1
