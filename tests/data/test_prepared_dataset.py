import pytest

from nlp_platform.data.metadata import DatasetMetadata
from nlp_platform.data.prepared_dataset import PreparedDataset
from nlp_platform.data.records import TextRecord


@pytest.fixture
def records() -> list[TextRecord]:
    return [
        TextRecord(
            text="hello world",
            label=0,
        ),
        TextRecord(
            text="machine learning",
            label=1,
        ),
        TextRecord(
            text="natural language",
            label=1,
        ),
    ]


@pytest.fixture
def metadata() -> DatasetMetadata:
    return DatasetMetadata(
        name="test-dataset",
        version="1.0.0",
        source="unit-test",
        created_at="2026-01-01T00:00:00+00:00",
        num_records=3,
        num_classes=2,
    )


def test_create_prepared_dataset(
    records: list[TextRecord],
    metadata: DatasetMetadata,
):
    dataset = PreparedDataset.create(
        records,
        metadata,
    )

    assert dataset.records == tuple(records)
    assert dataset.metadata == metadata


def test_distribution_is_calculated(
    records: list[TextRecord],
    metadata: DatasetMetadata,
):
    dataset = PreparedDataset.create(
        records,
        metadata,
    )

    assert dataset.distribution.counts == {
        0: 1,
        1: 2,
    }

    assert dataset.distribution.total_records == 3
    assert dataset.distribution.num_classes == 2


def test_size(
    records: list[TextRecord],
    metadata: DatasetMetadata,
):
    dataset = PreparedDataset.create(
        records,
        metadata,
    )

    assert dataset.size == 3


def test_labels(
    records: list[TextRecord],
    metadata: DatasetMetadata,
):
    dataset = PreparedDataset.create(
        records,
        metadata,
    )

    assert dataset.labels == (
        0,
        1,
        1,
    )


def test_texts(
    records: list[TextRecord],
    metadata: DatasetMetadata,
):
    dataset = PreparedDataset.create(
        records,
        metadata,
    )

    assert dataset.texts == (
        "hello world",
        "machine learning",
        "natural language",
    )


def test_records_are_stored_as_tuple(
    records: list[TextRecord],
    metadata: DatasetMetadata,
):
    dataset = PreparedDataset.create(
        records,
        metadata,
    )

    assert isinstance(dataset.records, tuple)


def test_metadata_record_count_must_match(
    records: list[TextRecord],
):
    metadata = DatasetMetadata(
        name="test-dataset",
        version="1.0.0",
        source="unit-test",
        created_at="2026-01-01T00:00:00+00:00",
        num_records=10,
        num_classes=2,
    )

    with pytest.raises(
        ValueError,
        match="num_records does not match",
    ):
        PreparedDataset.create(
            records,
            metadata,
        )


def test_metadata_class_count_must_match(
    records: list[TextRecord],
):
    metadata = DatasetMetadata(
        name="test-dataset",
        version="1.0.0",
        source="unit-test",
        created_at="2026-01-01T00:00:00+00:00",
        num_records=3,
        num_classes=4,
    )

    with pytest.raises(
        ValueError,
        match="num_classes does not match",
    ):
        PreparedDataset.create(
            records,
            metadata,
        )