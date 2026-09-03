from dataclasses import dataclass


@dataclass(frozen=True)
class LabelMapper:
    """Map integer class IDs to human-readable labels."""

    labels: dict[int, str]

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("labels must not be empty")

        if any(not isinstance(key, int) for key in self.labels):
            raise TypeError("label IDs must be integers")

        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.labels.values()
        ):
            raise ValueError("label names must be non-empty strings")

    def to_name(self, label: int) -> str:
        """Convert a label ID to its name."""

        try:
            return self.labels[label]
        except KeyError as exc:
            raise ValueError(
                f"unknown label ID: {label}"
            ) from exc

    def to_id(self, name: str) -> int:
        """Convert a label name to its ID."""

        for label_id, label_name in self.labels.items():
            if label_name == name:
                return label_id

        raise ValueError(
            f"unknown label name: {name}"
        )

    @property
    def num_classes(self) -> int:
        """Return the number of configured classes."""

        return len(self.labels)