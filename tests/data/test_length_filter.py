import pytest

from nlp_platform.data.length_filter import TextLengthFilter
from nlp_platform.data.records import TextRecord


def test_accepts_text_within_range():
    record = TextRecord(text="hello world", label=0)

    text_filter = TextLengthFilter(
        min_length=5,
        max_length=20,
    )

    assert text_filter.accepts(record)


def test_rejects_text_below_minimum():
    record = TextRecord(text="hi", label=0)

    text_filter = TextLengthFilter(min_length=5)

    assert not text_filter.accepts(record)


def test_rejects_text_above_maximum():
    record = TextRecord(text="hello world", label=0)

    text_filter = TextLengthFilter(max_length=5)

    assert not text_filter.accepts(record)


def test_max_length_is_optional():
    record = TextRecord(text="a very long text", label=0)

    text_filter = TextLengthFilter(min_length=5)

    assert text_filter.accepts(record)


def test_filter_removes_invalid_records():
    records = [
        TextRecord(text="hi", label=0),
        TextRecord(text="hello", label=1),
        TextRecord(text="hello world", label=2),
        TextRecord(text="this text is too long", label=3),
    ]

    text_filter = TextLengthFilter(
        min_length=5,
        max_length=11,
    )

    result = text_filter.filter(records)

    assert result == [
        TextRecord(text="hello", label=1),
        TextRecord(text="hello world", label=2),
    ]


def test_filter_preserves_order():
    records = [
        TextRecord(text="first", label=0),
        TextRecord(text="second", label=1),
        TextRecord(text="third", label=2),
    ]

    text_filter = TextLengthFilter(min_length=5)

    result = text_filter.filter(records)

    assert [record.text for record in result] == [
        "first",
        "second",
        "third",
    ]


def test_negative_min_length_is_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        TextLengthFilter(min_length=-1)


def test_invalid_length_range_is_rejected():
    with pytest.raises(
        ValueError,
        match="greater than or equal",
    ):
        TextLengthFilter(
            min_length=10,
            max_length=5,
        )