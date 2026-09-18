import json
from pathlib import Path

from nlp_platform.ml.experiment_result import ExperimentResult
from nlp_platform.ml.training_history import EpochMetrics, TrainingHistory


class ExperimentResultPersistence:
    """Persist experiment results as JSON."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        result: ExperimentResult,
        filename: str | Path | None = None,
    ) -> Path:
        output_path = Path(filename or f"{result.experiment_name}.json")

        if not output_path.is_absolute():
            output_path = self._directory / output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "experiment_name": result.experiment_name,
            "seed": result.seed,
            "epochs_completed": result.epochs_completed,
            "stopped_early": result.stopped_early,
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "best_checkpoint": result.best_checkpoint,
            "history": [
                {
                    "epoch": metrics.epoch,
                    "train_loss": metrics.train_loss,
                    "validation_loss": metrics.validation_loss,
                    "validation_accuracy": metrics.validation_accuracy,
                }
                for metrics in result.history.epochs
            ],
        }

        output_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        return output_path

    def load(self, path: str | Path) -> ExperimentResult:
        input_path = Path(path)

        if not input_path.is_absolute():
            input_path = self._directory / input_path

        if not input_path.exists():
            raise FileNotFoundError(f"Experiment result not found: {input_path}")

        payload = json.loads(input_path.read_text(encoding="utf-8"))

        history = TrainingHistory(
            epochs=[
                EpochMetrics(
                    epoch=item["epoch"],
                    train_loss=item["train_loss"],
                    validation_loss=item["validation_loss"],
                    validation_accuracy=item["validation_accuracy"],
                )
                for item in payload["history"]
            ]
        )

        return ExperimentResult(
            experiment_name=payload["experiment_name"],
            seed=payload["seed"],
            history=history,
            epochs_completed=payload["epochs_completed"],
            stopped_early=payload["stopped_early"],
            best_epoch=payload["best_epoch"],
            best_validation_loss=payload["best_validation_loss"],
            best_checkpoint=payload["best_checkpoint"],
        )