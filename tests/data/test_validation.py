import pytest

from nlp_platform.data.records import TextRecord
from nlp_platform.data.validation import (
    DataValidationError,
    validate_record,
    validate_records,
)


def test_valid_record_passes_validation():
    record = TextRecord(
        text="The market increased today.",
        label=1,
    )

    validate_record(record)


def test_non_record_is_rejected():
    with pytest.raises(DataValidationError):
        validate_record("invalid")


def test_negative_label_is_rejected():
    record = TextRecord(
        text="The market declined today.",
        label=-1,
    )

    with pytest.raises(DataValidationError):
        validate_record(record)


def test_multiple_valid_records_pass_validation():
    records = [
        TextRecord(text="Technology stocks rose.", label=0),
        TextRecord(text="Sports news dominates.", label=1),
        TextRecord(text="Political debate continues.", label=2),
    ]

    validate_records(records)


def test_invalid_record_reports_index():
    records = [
        TextRecord(text="Valid record.", label=0),
        TextRecord(text="Another valid record.", label=1),
        TextRecord(text="Invalid label.", label=-1),
    ]

    with pytest.raises(DataValidationError, match="index 2"):
        validate_records(records)


def test_empty_collection_passes_validation():
    validate_records([])