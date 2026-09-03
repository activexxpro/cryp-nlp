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