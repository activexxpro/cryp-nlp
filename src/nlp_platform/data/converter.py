from collections.abc import Iterable, Mapping

from nlp_platform.data.records import TextRecord


def row_to_record(row: Mapping[str, object]) -> TextRecord:
    """Convert a dataset row into a validated TextRecord."""

    if "text" not in row:
        raise KeyError("dataset row is missing 'text'")

    if "label" not in row:
        raise KeyError("dataset row is missing 'label'")

    return TextRecord(
        text=row["text"],
        label=row["label"],
    )


def rows_to_records(
    rows: Iterable[Mapping[str, object]],
) -> list[TextRecord]:
    """Convert multiple dataset rows into TextRecord objects."""

    return [row_to_record(row) for row in rows]