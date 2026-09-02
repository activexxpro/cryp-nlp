import pytest

from nlp_platform.data.converter import (
    row_to_record,
    rows_to_records,
)
from nlp_platform.data.records import TextRecord


def test_row_to_record():
    row = {
        "text": "Technology stocks increased.",
        "label": 0,
    }

    result = row_to_record(row)

    assert result == TextRecord(
        text="Technology stocks increased.",
        label=0,
    )


def test_missing_text_column_is_rejected():
    row = {
        "label": 1,
    }

    with pytest.raises(KeyError, match="text"):
        row_to_record(row)


def test_missing_label_column_is_rejected():
    row = {
        "text": "Some news article.",
    }

    with pytest.raises(KeyError, match="label"):
        row_to_record(row)


def test_rows_to_records():
    rows = [
        {
            "text": "Technology news.",
            "label": 0,
        },
        {
            "text": "Sports news.",
            "label": 1,
        },
        {
            "text": "Business news.",
            "label": 2,
        },
    ]

    result = rows_to_records(rows)

    assert result == [
        TextRecord(text="Technology news.", label=0),
        TextRecord(text="Sports news.", label=1),
        TextRecord(text="Business news.", label=2),
    ]


def test_rows_to_records_returns_list():
    rows = [
        {
            "text": "Example article.",
            "label": 0,
        }
    ]

    result = rows_to_records(rows)

    assert isinstance(result, list)
    assert all(isinstance(record, TextRecord) for record in result)