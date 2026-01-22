from train.stats import EpochStats


class EarlyStopper:
    def __init__(self, patience: int):
        self._patience = patience
        self._best_epoch = 0
        self._best_loss = float("inf")
        self._counter = 0
        self._improved = False

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def counter(self) -> int:
        return self._counter

    @property
    def patience(self) -> int:
        return self._patience

    @property
    def should_save_state(self) -> bool:
        return self._improved

    @property
    def epochs_since_improvement(self) -> int:
        return self._counter

    def should_stop(self, val_epoch_stats: EpochStats) -> bool:
        self._improved = False

        if val_epoch_stats.loss < self._best_loss:
            self._best_loss = val_epoch_stats.loss
            self._best_epoch = val_epoch_stats.epoch
            self._counter = 0
            self._improved = True
        else:
            self._counter += 1

        return self._counter >= self._patience
