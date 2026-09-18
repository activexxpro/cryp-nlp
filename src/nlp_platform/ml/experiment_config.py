from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration describing one model training experiment."""

    name: str = "baseline"
    seed: int = 42
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "name must not be empty"
            )

        if self.seed < 0:
            raise ValueError(
                "seed must not be negative"
            )

    @property
    def identifier(self) -> str:
        """Return a normalized experiment identifier."""

        return self.name.strip().lower().replace(
            " ",
            "-",
        )