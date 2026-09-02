import pytest

from nlp_platform.data.batch import prepare_records
from nlp_platform.data.validation import DataValidationError


def test_prepare_records_converts_and_validates_rows():
    rows = [
        {
            "text": "Technology stocks increased.",
            "label": 0,
        },
        {
            "text": "The team won the championship.",
            "label": 1,
        },
    ]

    records = prepare_records(rows)

    assert len(records) == 2
    assert records[0].text == "Technology stocks increased."
    assert records[0].label == 0
    assert records[1].text == "The team won the championship."
    assert records[1].label == 1


def test_prepare_records_rejects_invalid_label():
    rows = [
        {
            "text": "Valid article.",
            "label": 0,
        },
        {
            "text": "Invalid article.",
            "label": -1,
        },
    ]

    with pytest.raises(DataValidationError, match="index 1"):
        prepare_records(rows)


def test_prepare_records_rejects_missing_text():
    rows = [
        {
            "label": 0,
        }
    ]

    with pytest.raises(KeyError, match="text"):
        prepare_records(rows)


def test_prepare_records_rejects_missing_label():
    rows = [
        {
            "text": "Article without a label.",
        }
    ]

    with pytest.raises(KeyError, match="label"):
        prepare_records(rows)


def test_prepare_empty_dataset():
    records = prepare_records([])

    assert records == []