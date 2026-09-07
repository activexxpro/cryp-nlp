from nlp_platform.data.preparation import (
    DatasetPreparationService,
)
from nlp_platform.data.records import TextRecord


def test_prepare_dataset():
    records = [
        TextRecord(
            text="  HELLO    WORLD  ",
            label=0,
        ),
        TextRecord(
            text="Machine   Learning",
            label=1,
        ),
    ]

    service = DatasetPreparationService()

    result = service.prepare(records)

    assert result == [
        TextRecord(
            text="hello world",
            label=0,
        ),
        TextRecord(
            text="machine learning",
            label=1,
        ),
    ]


def test_prepare_empty_dataset():
    service = DatasetPreparationService()

    assert service.prepare([]) == []


def test_prepare_removes_duplicates():
    records = [
        TextRecord(
            text="Hello",
            label=0,
        ),
        TextRecord(
            text="hello",
            label=0,
        ),
    ]

    service = DatasetPreparationService()

    result = service.prepare(records)

    assert result == [
        TextRecord(
            text="hello",
            label=0,
        ),
    ]


def test_prepare_preserves_labels():
    records = [
        TextRecord(
            text="Sports",
            label=1,
        ),
        TextRecord(
            text="Business",
            label=2,
        ),
        TextRecord(
            text="Science",
            label=3,
        ),
    ]

    service = DatasetPreparationService()

    result = service.prepare(records)

    assert [record.label for record in result] == [
        1,
        2,
        3,
    ]