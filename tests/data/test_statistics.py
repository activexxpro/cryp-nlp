from nlp_platform.data.records import TextRecord
from nlp_platform.data.statistics import (
    RecordStatistics,
    calculate_statistics,
)


def test_calculate_statistics():
    records = [
        TextRecord(text="Hello", label=0),
        TextRecord(text="World!", label=0),
        TextRecord(text="NLP platform", label=1),
        TextRecord(text="Machine learning", label=1),
        TextRecord(text="Python", label=2),
    ]

    stats = calculate_statistics(records)

    assert stats.total_records == 5
    assert stats.num_classes == 3
    assert stats.label_counts == {
        0: 2,
        1: 2,
        2: 1,
    }
    assert stats.average_text_length == 9
    assert stats.min_text_length == 5
    assert stats.max_text_length == 16


def test_empty_records():
    stats = calculate_statistics([])

    assert stats == RecordStatistics(
        total_records=0,
        num_classes=0,
        label_counts={},
        average_text_length=0.0,
        min_text_length=0,
        max_text_length=0,
    )


def test_single_record():
    records = [
        TextRecord(
            text="Natural language processing",
            label=3,
        )
    ]

    stats = calculate_statistics(records)

    assert stats.total_records == 1
    assert stats.num_classes == 1
    assert stats.label_counts == {3: 1}
    assert stats.average_text_length == 27
    assert stats.min_text_length == 27
    assert stats.max_text_length == 27


def test_label_counts_are_sorted():
    records = [
        TextRecord(text="A", label=3),
        TextRecord(text="B", label=1),
        TextRecord(text="C", label=2),
        TextRecord(text="D", label=1),
    ]

    stats = calculate_statistics(records)

    assert list(stats.label_counts.keys()) == [1, 2, 3]