from datetime import datetime

import pytest

from nlp_platform.data.metadata import DatasetMetadata


def test_create_metadata():
    metadata = DatasetMetadata.create(
        name="ag-news",
        version="1.0.0",
        source="huggingface",
        num_records=1000,
        num_classes=4,
    )

    assert metadata.name == "ag-news"
    assert metadata.version == "1.0.0"
    assert metadata.source == "huggingface"
    assert metadata.num_records == 1000
    assert metadata.num_classes == 4

    timestamp = datetime.fromisoformat(
        metadata.created_at
    )

    assert timestamp.tzinfo is not None


def test_metadata_is_immutable():
    metadata = DatasetMetadata.create(
        name="dataset",
        version="1.0.0",
        source="local",
        num_records=10,
        num_classes=2,
    )

    with pytest.raises(AttributeError):
        metadata.name = "changed"


def test_empty_name_is_rejected():
    with pytest.raises(
        ValueError,
        match="name must not be empty",
    ):
        DatasetMetadata.create(
            name="",
            version="1.0.0",
            source="local",
            num_records=10,
            num_classes=2,
        )


def test_empty_version_is_rejected():
    with pytest.raises(
        ValueError,
        match="version must not be empty",
    ):
        DatasetMetadata.create(
            name="dataset",
            version="",
            source="local",
            num_records=10,
            num_classes=2,
        )


def test_empty_source_is_rejected():
    with pytest.raises(
        ValueError,
        match="source must not be empty",
    ):
        DatasetMetadata.create(
            name="dataset",
            version="1.0.0",
            source="",
            num_records=10,
            num_classes=2,
        )


def test_negative_record_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="num_records must not be negative",
    ):
        DatasetMetadata.create(
            name="dataset",
            version="1.0.0",
            source="local",
            num_records=-1,
            num_classes=2,
        )


def test_negative_class_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="num_classes must not be negative",
    ):
        DatasetMetadata.create(
            name="dataset",
            version="1.0.0",
            source="local",
            num_records=10,
            num_classes=-1,
        )


def test_metadata_equality():
    metadata = DatasetMetadata(
        name="dataset",
        version="1.0.0",
        source="local",
        created_at="2026-01-01T00:00:00+00:00",
        num_records=10,
        num_classes=2,
    )

    same_metadata = DatasetMetadata(
        name="dataset",
        version="1.0.0",
        source="local",
        created_at="2026-01-01T00:00:00+00:00",
        num_records=10,
        num_classes=2,
    )

    assert metadata == same_metadata