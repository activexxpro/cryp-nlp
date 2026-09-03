from nlp_platform.data.duplicates import (
    find_duplicate_indices,
    remove_duplicates,
)
from nlp_platform.data.records import TextRecord


def test_find_duplicate_indices():
    records = [
        TextRecord(text="hello", label=0),
        TextRecord(text="world", label=1),
        TextRecord(text="hello", label=0),
        TextRecord(text="hello", label=2),
    ]

    duplicates = find_duplicate_indices(records)

    assert duplicates == {
        "hello": [0, 2, 3],
    }


def test_no_duplicates():
    records = [
        TextRecord(text="hello", label=0),
        TextRecord(text="world", label=1),
    ]

    assert find_duplicate_indices(records) == {}


def test_remove_duplicates_preserves_first_occurrence():
    records = [
        TextRecord(text="hello", label=0),
        TextRecord(text="world", label=1),
        TextRecord(text="hello", label=0),
    ]

    result = remove_duplicates(records)

    assert result == [
        TextRecord(text="hello", label=0),
        TextRecord(text="world", label=1),
    ]


def test_remove_duplicates_preserves_order():
    records = [
        TextRecord(text="a", label=0),
        TextRecord(text="b", label=1),
        TextRecord(text="c", label=2),
        TextRecord(text="b", label=1),
        TextRecord(text="a", label=0),
    ]

    result = remove_duplicates(records)

    assert [record.text for record in result] == [
        "a",
        "b",
        "c",
    ]


def test_empty_records():
    assert find_duplicate_indices([]) == {}
    assert remove_duplicates([]) == []