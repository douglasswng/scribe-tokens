from pathlib import Path
from typing import Protocol

import mlflow
import numpy as np
import swanlab
from PIL import Image

from utils.distributed_context import distributed_context

type ImageType = np.ndarray | Image.Image


class Tracker(Protocol):
    def begin_experiment(self, name: str, artifact_dir: str | Path) -> None: ...
    def begin_run(self, tags: list[str], run_name: str) -> None: ...
    def log_params(self, params: dict[str, float]) -> None: ...
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    def log_image(self, image: ImageType, name: str, caption: str | None = None) -> None: ...
    def is_active(self) -> bool: ...
    def end_run(self) -> None: ...


class MLFlowTracker(Tracker):
    def __init__(self):
        self.step_counters: dict[str, int] = {}

    def begin_experiment(self, name: str, artifact_dir: str | Path) -> None:
        if distributed_context.is_worker:
            return

        mlflow.set_tracking_uri(artifact_dir)
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is None:
            mlflow.create_experiment(name=name)
        mlflow.set_experiment(name)

    def begin_run(self, tags: list[str], run_name: str) -> None:
        if distributed_context.is_worker:
            return

        if self.is_active():
            self.end_run()

        mlflow.start_run(run_name=run_name, tags={tag: tag for tag in tags})
        self.step_counters.clear()

    def log_params(self, params: dict[str, float]) -> None:
        if distributed_context.is_worker:
            return

        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if distributed_context.is_worker:
            return

        for metric_name, value in metrics.items():
            current_step = self.step_counters.get(metric_name, 0)
            mlflow.log_metric(metric_name, value, step=current_step)
            self.step_counters[metric_name] = current_step + 1

    def log_image(self, image: ImageType, name: str, caption: str | None = None) -> None:
        if distributed_context.is_worker:
            return

        mlflow.log_image(image, f"{name}: {caption or ''}")

    def end_run(self) -> None:
        if distributed_context.is_worker:
            return

        mlflow.end_run()

    def is_active(self) -> bool:
        return mlflow.active_run() is not None


class SwanLabTracker(Tracker):
    def __init__(self):
        self.tracking_uri: str | Path | None = None
        self.experiment_name: str | None = None

    def begin_experiment(self, name: str, artifact_dir: str | Path) -> None:
        if distributed_context.is_worker:
            return

        self.tracking_uri = artifact_dir
        self.experiment_name = name

    def begin_run(self, tags: list[str], run_name: str) -> None:
        if distributed_context.is_worker:
            return

        if self.experiment_name is None or self.tracking_uri is None:
            raise ValueError(
                "Experiment name or tracking uri is not set, call begin_experiment() first"
            )

        if self.is_active():
            self.end_run()

        swanlab.init(
            project=self.experiment_name,
            experiment_name=run_name,
            tags=tags,
            logdir=str(self.tracking_uri),
        )

    def log_params(self, params: dict[str, float]) -> None:
        if distributed_context.is_worker:
            return

        if swanlab.config is None:
            raise RuntimeError("SwanLab is not initialized. Call begin_run() first.")

        for key, value in params.items():
            swanlab.config[key] = value

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if distributed_context.is_worker:
            return

        swanlab.log(metrics)

    def log_image(self, image: ImageType, name: str, caption: str | None = None) -> None:
        if distributed_context.is_worker:
            return

        swanlab.log({name: swanlab.Image(image, caption=caption or name)})

    def end_run(self) -> None:
        if distributed_context.is_worker:
            return

        swanlab.finish()

    def is_active(self) -> bool:
        return swanlab.run.get_run() is not None
