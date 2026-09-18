import json

import pytest

from nlp_platform.ml.experiment_persistence import ExperimentResultPersistence
from nlp_platform.ml.experiment_result import ExperimentResult
from nlp_platform.ml.training_history import EpochMetrics, TrainingHistory


def create_result() -> ExperimentResult:
    history = TrainingHistory(
        epochs=[
            EpochMetrics(
                epoch=1,
                train_loss=0.8,
                validation_loss=0.7,
                validation_accuracy=0.75,
            ),
            EpochMetrics(
                epoch=2,
                train_loss=0.6,
                validation_loss=0.5,
                validation_accuracy=0.85,
            ),
        ]
    )

    return ExperimentResult(
        experiment_name="baseline",
        seed=42,
        history=history,
        epochs_completed=2,
        stopped_early=False,
        best_epoch=2,
        best_validation_loss=0.5,
        best_checkpoint="checkpoints/best_model.pt",
    )


def test_save_creates_json_file(tmp_path) -> None:
    persistence = ExperimentResultPersistence(tmp_path)

    path = persistence.save(create_result())

    assert path.exists()
    assert path.name == "baseline.json"


def test_save_writes_valid_json(tmp_path) -> None:
    persistence = ExperimentResultPersistence(tmp_path)

    path = persistence.save(create_result())

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["experiment_name"] == "baseline"
    assert payload["seed"] == 42
    assert payload["epochs_completed"] == 2
    assert payload["stopped_early"] is False
    assert payload["best_epoch"] == 2
    assert payload["best_validation_loss"] == 0.5
    assert len(payload["history"]) == 2


def test_load_restores_experiment_result(tmp_path) -> None:
    persistence = ExperimentResultPersistence(tmp_path)
    original = create_result()

    path = persistence.save(original)
    restored = persistence.load(path)

    assert isinstance(restored, ExperimentResult)
    assert restored.experiment_name == original.experiment_name
    assert restored.seed == original.seed
    assert restored.epochs_completed == original.epochs_completed
    assert restored.stopped_early == original.stopped_early
    assert restored.best_epoch == original.best_epoch
    assert restored.best_validation_loss == original.best_validation_loss
    assert restored.best_checkpoint == original.best_checkpoint


def test_load_restores_training_history(tmp_path) -> None:
    persistence = ExperimentResultPersistence(tmp_path)
    original = create_result()

    path = persistence.save(original)
    restored = persistence.load(path)

    assert len(restored.history.epochs) == 2

    assert restored.history.epochs[0] == EpochMetrics(
        epoch=1,
        train_loss=0.8,
        validation_loss=0.7,
        validation_accuracy=0.75,
    )

    assert restored.history.epochs[1] == EpochMetrics(
        epoch=2,
        train_loss=0.6,
        validation_loss=0.5,
        validation_accuracy=0.85,
    )


def test_save_supports_custom_filename(tmp_path) -> None:
    persistence = ExperimentResultPersistence(tmp_path)

    path = persistence.save(
        create_result(),
        filename="experiment-001.json",
    )

    assert path == tmp_path / "experiment-001.json"
    assert path.exists()


def test_load_missing_file_raises_error(tmp_path) -> None:
    persistence = ExperimentResultPersistence(tmp_path)

    with pytest.raises(FileNotFoundError):
        persistence.load("missing.json")