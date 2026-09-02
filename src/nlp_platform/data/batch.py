from collections.abc import Iterable, Mapping

from nlp_platform.data.converter import rows_to_records
from nlp_platform.data.records import TextRecord
from nlp_platform.data.validation import validate_records


def prepare_records(
    rows: Iterable[Mapping[str, object]],
) -> list[TextRecord]:
    """Convert dataset rows and validate the resulting records."""

    records = rows_to_records(rows)
    validate_records(records)

    return records