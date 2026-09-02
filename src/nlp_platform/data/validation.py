from collections.abc import Iterable

from nlp_platform.data.records import TextRecord


class DataValidationError(ValueError):
    """Raised when one or more text records fail validation."""


def validate_record(record: TextRecord) -> None:
    """Validate a single text record."""

    if not isinstance(record, TextRecord):
        raise DataValidationError(
            "record must be an instance of TextRecord"
        )

    if not record.text.strip():
        raise DataValidationError(
            "record text must not be empty"
        )

    if record.label < 0:
        raise DataValidationError(
            "record label must not be negative"
        )


def validate_records(records: Iterable[TextRecord]) -> None:
    """Validate a collection of text records."""

    for index, record in enumerate(records):
        try:
            validate_record(record)
        except DataValidationError as exc:
            raise DataValidationError(
                f"invalid record at index {index}: {exc}"
            ) from exc