from collections.abc import Sequence

from nlp_platform.data.records import TextRecord


class TextLengthFilter:
    """Filter text records by character length."""

    def __init__(
        self,
        min_length: int = 1,
        max_length: int | None = None,
    ) -> None:
        if min_length < 0:
            raise ValueError("min_length must not be negative")

        if max_length is not None and max_length < min_length:
            raise ValueError(
                "max_length must be greater than or equal to min_length"
            )

        self._min_length = min_length
        self._max_length = max_length

    def accepts(self, record: TextRecord) -> bool:
        """Return whether a record satisfies the length constraints."""

        length = len(record.text)

        if length < self._min_length:
            return False

        return not (
            self._max_length is not None
            and length > self._max_length
        )

    def filter(
        self,
        records: Sequence[TextRecord],
    ) -> list[TextRecord]:
        """Return records that satisfy the length constraints."""

        return [
            record
            for record in records
            if self.accepts(record)
        ]