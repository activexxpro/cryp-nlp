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
def test_prepare_removes_duplicates():
    pipeline = DatasetPreparationPipeline()

    records = [
        TextRecord(
            text=f"Document class zero {index}",
            label=0,
        )
        for index in range(10)
    ] + [
        TextRecord(
            text=f"Document class one {index}",
            label=1,
        )
        for index in range(10)
    ]

    records.extend(
        [
            TextRecord(
                text="Duplicate document",
                label=0,
            ),
            TextRecord(
                text="Duplicate document",
                label=0,
            ),
        ]
    )

    splits = pipeline.prepare(records)

    prepared = (
        splits.train
        + splits.validation
        + splits.test
    )

    texts = [record.text for record in prepared]

    assert texts.count("duplicate document") == 1
    assert len(texts) == len(set(texts))

def test_prepare_normalizes_before_duplicate_detection():
    pipeline = DatasetPreparationPipeline()

    records = [
        TextRecord(
            text=f"Class zero document {index}",
            label=0,
        )
        for index in range(10)
    ] + [
        TextRecord(
            text=f"Class one document {index}",
            label=1,
        )
        for index in range(10)
    ]

    records.extend(
        [
            TextRecord(
                text="  Hello    World  ",
                label=0,
            ),
            TextRecord(
                text="hello world",
                label=0,
            ),
        ]
    )

    splits = pipeline.prepare(records)

    prepared = (
        splits.train
        + splits.validation
        + splits.test
    )

    texts = [record.text for record in prepared]

    assert texts.count("hello world") == 1

    
def test_keep_duplicates_configuration():
    pipeline = DatasetPreparationPipeline(
        remove_duplicate_records=False,
    )

    records = [
        TextRecord(
            text="Example document",
            label=0,
        )
        for _ in range(10)
    ] + [
        TextRecord(
            text="Another document",
            label=1,
        )
        for _ in range(10)
    ]

    splits = pipeline.prepare(records)

    prepared = (
        splits.train
        + splits.validation
        + splits.test
    )

    assert len(prepared) == 20


def test_prepare_applies_length_filter():
    pipeline = DatasetPreparationPipeline(
        min_length=10,
        max_length=30,
    )

    records = [
        TextRecord(
            text=f"Document with sufficient length {index}",
            label=index % 2,
        )
        for index in range(20)
    ]

    splits = pipeline.prepare(records)

    prepared = (
        splits.train
        + splits.validation
        + splits.test
    )

    assert all(
        10 <= len(record.text) <= 30
        for record in prepared
    )


def test_prepare_preserves_all_labels():
    pipeline = DatasetPreparationPipeline()

    records = [
        TextRecord(
            text=f"Document number {index}",
            label=index % 2,
        )
        for index in range(100)
    ]

    splits = pipeline.prepare(records)

    prepared = (
        splits.train
        + splits.validation
        + splits.test
    )

    assert [record.label for record in prepared].count(0) == 50
    assert [record.label for record in prepared].count(1) == 50


def test_build_metadata_matches_processed_dataset():
    pipeline = DatasetPreparationPipeline()

    records = [
        TextRecord(
            text=f"Document number {index}",
            label=index % 2,
        )
        for index in range(20)
    ]

    dataset = pipeline.build(
        records,
        name="test-dataset",
        version="1.0.0",
        source="unit-test",
    )

    assert dataset.metadata.num_records == dataset.size
    assert (
        dataset.metadata.num_classes
        == dataset.distribution.num_classes
    )


def test_build_uses_processed_records():
    pipeline = DatasetPreparationPipeline()

    records = [
        TextRecord(
            text=f"  DOCUMENT {index}    TEXT  ",
            label=index % 2,
        )
        for index in range(20)
    ]

    dataset = pipeline.build(records)

    assert all(
        record.text == record.text.lower()
        for record in dataset.records
    )

    assert all(
        "  " not in record.text
        for record in dataset.records
    )