from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata describing a prepared dataset."""

    name: str
    version: str
    source: str
    created_at: str
    num_records: int
    num_classes: int

    @classmethod
    def create(
        cls,
        name: str,
        version: str,
        source: str,
        num_records: int,
        num_classes: int,
    ) -> "DatasetMetadata":
        """Create metadata with the current UTC timestamp."""

        if not name.strip():
            raise ValueError("name must not be empty")

        if not version.strip():
            raise ValueError("version must not be empty")

        if not source.strip():
            raise ValueError("source must not be empty")

        if num_records < 0:
            raise ValueError("num_records must not be negative")

        if num_classes < 0:
            raise ValueError("num_classes must not be negative")

        return cls(
            name=name,
            version=version,
            source=source,
            created_at=datetime.now(timezone.utc).isoformat(),
            num_records=num_records,
            num_classes=num_classes,
        )