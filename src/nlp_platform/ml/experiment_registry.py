from pathlib import Path

from nlp_platform.ml.experiment_persistence import ExperimentResultPersistence
from nlp_platform.ml.experiment_result import ExperimentResult


class ExperimentRegistry:
    """Discover and load persisted experiment results."""

    def __init__(
        self,
        directory: str | Path,
        persistence: ExperimentResultPersistence | None = None,
    ) -> None:
        self._directory = Path(directory)
        self._persistence = persistence or ExperimentResultPersistence(
            self._directory,
        )

    def list_experiments(self) -> list[str]:
        """Return experiment result filenames in deterministic order."""
        if not self._directory.exists():
            return []

        return sorted(
            path.name
            for path in self._directory.glob("*.json")
            if path.is_file()
        )

    def load_experiment(self, filename: str) -> ExperimentResult:
        """Load one experiment result by filename."""
        return self._persistence.load(filename)

    def load_all(self) -> list[ExperimentResult]:
        """Load all persisted experiments in deterministic order."""
        return [
            self.load_experiment(filename)
            for filename in self.list_experiments()
        ]

    @property
    def count(self) -> int:
        """Return the number of persisted experiments."""
        return len(self.list_experiments())