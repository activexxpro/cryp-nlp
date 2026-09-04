from nlp_platform.data.dataset_preparation import (
    DatasetPreparationPipeline,
)
from nlp_platform.data.records import TextRecord
from nlp_platform.data.split_config import SplitConfig


def make_records(count: int = 100) -> list[TextRecord]:
    return [
        TextRecord(
            text=f"Example document number {index}",
            label=index % 2,
        )
        for index in range(count)
    ]


def test_prepare_returns_dataset_splits():
    pipeline = DatasetPreparationPipeline()

    records = make_records()

    splits = pipeline.prepare(records)

    assert len(splits.train) == 80
    assert len(splits.validation) == 10
    assert len(splits.test) == 10


def test_prepare_applies_preprocessing():
    pipeline = DatasetPreparationPipeline()

    records = [
        TextRecord(
            text=f"  HELLO    WORLD {index}  ",
            label=index % 2,
        )
        for index in range(20)
    ]

    splits = pipeline.prepare(records)

    all_records = (
        splits.train
        + splits.validation
        + splits.test
    )

    assert all_records
    assert all(
        "  " not in record.text
        for record in all_records
    )
    assert all(
        record.text == record.text.lower()
        for record in all_records
    )

    
def test_prepare_preserves_labels():
    pipeline = DatasetPreparationPipeline()

    records = make_records()

    splits = pipeline.prepare(records)

    prepared = (
        splits.train
        + splits.validation
        + splits.test
    )

    assert {record.label for record in prepared} == {
        0,
        1,
    }


def test_custom_split_configuration():
    config = SplitConfig(
        validation_size=0.2,
        test_size=0.2,
        seed=42,
    )

    pipeline = DatasetPreparationPipeline(
        split_config=config,
    )

    splits = pipeline.prepare(
        make_records(),
    )

    assert len(splits.train) == 60
    assert len(splits.validation) == 20
    assert len(splits.test) == 20


def test_create_metadata():
    pipeline = DatasetPreparationPipeline()

    records = make_records(20)

    metadata = pipeline.create_metadata(
        records,
        name="ag-news",
        version="1.0.0",
        source="huggingface",
    )

    assert metadata.name == "ag-news"
    assert metadata.version == "1.0.0"
    assert metadata.source == "huggingface"
    assert metadata.num_records == 20
    assert metadata.num_classes == 2


def test_build_prepared_dataset():
    pipeline = DatasetPreparationPipeline()

    records = make_records(20)

    dataset = pipeline.build(
        records,
        name="test-dataset",
        version="1.0.0",
        source="unit-test",
    )

    assert dataset.size == 20
    assert dataset.metadata.name == "test-dataset"
    assert dataset.metadata.num_records == 20
    assert dataset.metadata.num_classes == 2
    assert dataset.distribution.counts == {
        0: 10,
        1: 10,
    }