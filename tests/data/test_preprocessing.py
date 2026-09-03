from nlp_platform.data.preprocessing import (
    DatasetPreprocessingPipeline,
)
from nlp_platform.data.preprocessing_config import (
    PreprocessingConfig,
)
from nlp_platform.data.records import TextRecord


def test_preprocessing_pipeline():
    records = [
        TextRecord(
            text="  HELLO    WORLD  ",
            label=0,
        ),
        TextRecord(
            text="Python Programming",
            label=1,
        ),
    ]

    pipeline = DatasetPreprocessingPipeline()

    result = pipeline.process(records)

    assert result == [
        TextRecord(
            text="hello world",
            label=0,
        ),
        TextRecord(
            text="python programming",
            label=1,
        ),
    ]


def test_pipeline_removes_duplicates():
    records = [
        TextRecord(text="Hello", label=0),
        TextRecord(text="hello", label=0),
        TextRecord(text="World", label=1),
    ]

    pipeline = DatasetPreprocessingPipeline()

    result = pipeline.process(records)

    assert result == [
        TextRecord(text="hello", label=0),
        TextRecord(text="world", label=1),
    ]


def test_pipeline_filters_short_text():
    records = [
        TextRecord(text="Hi", label=0),
        TextRecord(text="Hello world", label=1),
    ]

    pipeline = DatasetPreprocessingPipeline(
        min_length=5,
    )

    result = pipeline.process(records)

    assert result == [
        TextRecord(
            text="hello world",
            label=1,
        ),
    ]


def test_pipeline_filters_long_text():
    records = [
        TextRecord(text="short", label=0),
        TextRecord(
            text="this is a very long sentence",
            label=1,
        ),
    ]

    pipeline = DatasetPreprocessingPipeline(
        max_length=10,
    )

    result = pipeline.process(records)

    assert result == [
        TextRecord(
            text="short",
            label=0,
        ),
    ]


def test_pipeline_can_keep_duplicates():
    records = [
        TextRecord(text="Hello", label=0),
        TextRecord(text="hello", label=0),
    ]

    pipeline = DatasetPreprocessingPipeline(
        remove_duplicate_records=False,
    )

    result = pipeline.process(records)

    assert result == [
        TextRecord(text="hello", label=0),
        TextRecord(text="hello", label=0),
    ]


def test_pipeline_respects_preprocessing_config():
    records = [
        TextRecord(
            text="  HELLO    WORLD  ",
            label=0,
        ),
    ]

    config = PreprocessingConfig(
        lowercase=False,
        strip_whitespace=False,
        normalize_whitespace=False,
    )

    pipeline = DatasetPreprocessingPipeline(
        config=config,
    )

    result = pipeline.process(records)

    assert result == records


def test_pipeline_handles_empty_collection():
    pipeline = DatasetPreprocessingPipeline()

    assert pipeline.process([]) == []


def test_pipeline_applies_length_filter_after_normalization():
    records = [
        TextRecord(
            text="   HI   ",
            label=0,
        ),
        TextRecord(
            text="   HELLO WORLD   ",
            label=1,
        ),
    ]

    pipeline = DatasetPreprocessingPipeline(
        min_length=5,
    )

    result = pipeline.process(records)

    assert result == [
        TextRecord(
            text="hello world",
            label=1,
        ),
    ]


def test_pipeline_deduplicates_after_normalization():
    records = [
        TextRecord(
            text="  Hello  ",
            label=0,
        ),
        TextRecord(
            text="hello",
            label=0,
        ),
        TextRecord(
            text="HELLO",
            label=0,
        ),
    ]

    pipeline = DatasetPreprocessingPipeline()

    result = pipeline.process(records)

    assert result == [
        TextRecord(
            text="hello",
            label=0,
        ),
    ]


def test_pipeline_preserves_labels():
    records = [
        TextRecord(
            text="Sports News",
            label=1,
        ),
        TextRecord(
            text="Business News",
            label=2,
        ),
    ]

    pipeline = DatasetPreprocessingPipeline()

    result = pipeline.process(records)

    assert [record.label for record in result] == [
        1,
        2,
    ]


def test_pipeline_preserves_record_order():
    records = [
        TextRecord(text="Third", label=2),
        TextRecord(text="First", label=0),
        TextRecord(text="Second", label=1),
    ]

    pipeline = DatasetPreprocessingPipeline()

    result = pipeline.process(records)

    assert [record.text for record in result] == [
        "third",
        "first",
        "second",
    ]