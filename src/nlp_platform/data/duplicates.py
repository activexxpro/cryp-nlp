from collections import defaultdict
from collections.abc import Sequence

from nlp_platform.data.records import TextRecord


def find_duplicate_indices(
    records: Sequence[TextRecord],
) -> dict[str, list[int]]:
    """Find duplicate texts and return their record indices."""

    indices: dict[str, list[int]] = defaultdict(list)

    for index, record in enumerate(records):
        indices[record.text].append(index)

    return {
        text: record_indices
        for text, record_indices in indices.items()
        if len(record_indices) > 1
    }


def remove_duplicates(
    records: Sequence[TextRecord],
) -> list[TextRecord]:
    """Remove duplicate texts while preserving the first occurrence."""

    seen: set[str] = set()
    unique_records: list[TextRecord] = []

    for record in records:
        if record.text in seen:
            continue

        seen.add(record.text)
        unique_records.append(record)

    return unique_records