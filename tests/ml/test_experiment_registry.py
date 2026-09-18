from pathlib import Path

from nlp_platform.ml.experiment_persistence import ExperimentResultPersistence
from nlp_platform.ml.experiment_registry import ExperimentRegistry
from nlp_platform.ml.experiment_result import ExperimentResult
from nlp_platform.ml.training_history import EpochMetrics, TrainingHistory


def create_result(name: str, accuracy: float) -> ExperimentResult:
    history = TrainingHistory(
        epochs=[
            EpochMetrics(
                epoch=1,
                train_loss=0.8,
                validation_loss=0.6,
                validation_accuracy=accuracy,
            ),
        ]
    )

    return ExperimentResult(
        experiment_name=name,
        seed=42,
        history=history,
        epochs_completed=1,
        stopped_early=False,
        best_epoch=1,
        best_validation_loss=0.6,
        best_checkpoint=None,
    )


def test_registry_lists_experiments(tmp_path: Path) -> None:
    persistence = ExperimentResultPersistence(tmp_path)

    persistence.save(create_result("experiment-a", 0.80))
    persistence.save(create_result("experiment-b", 0.90))

    registry = ExperimentRegistry(tmp_path)

    assert registry.list_experiments() == [
        "experiment-a.json",
        "experiment-b.json",
    ]


def test_registry_loads_experiment(tmp_path: Path) -> None:
    persistence = ExperimentResultPersistence(tmp_path)
    persistence.save(create_result("experiment-a", 0.85))

    registry = ExperimentRegistry(tmp_path)

    result = registry.load_experiment("experiment-a.json")

    assert isinstance(result, ExperimentResult)
    assert result.experiment_name == "experiment-a"
    assert result.final_validation_accuracy == 0.85


def test_registry_loads_all_experiments(tmp_path: Path) -> None:
    persistence = ExperimentResultPersistence(tmp_path)

    persistence.save(create_result("experiment-b", 0.90))
    persistence.save(create_result("experiment-a", 0.80))

    registry = ExperimentRegistry(tmp_path)

    results = registry.load_all()

    assert [result.experiment_name for result in results] == [
        "experiment-a",
        "experiment-b",
    ]


def test_registry_count(tmp_path: Path) -> None:
    persistence = ExperimentResultPersistence(tmp_path)

    persistence.save(create_result("experiment-a", 0.80))
    persistence.save(create_result("experiment-b", 0.90))

    registry = ExperimentRegistry(tmp_path)

    assert registry.count == 2


def test_empty_registry(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path)

    assert registry.list_experiments() == []
    assert registry.load_all() == []
    assert registry.count == 0