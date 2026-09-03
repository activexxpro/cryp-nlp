from nlp_platform.data.cleaning import (
    TextCleaningPipeline,
    lowercase_text,
)
from nlp_platform.data.preprocessor import TextPreprocessor
from nlp_platform.data.records import TextRecord


def test_process_record():
    pipeline = TextCleaningPipeline(
        transforms=[
            str.strip,
            lowercase_text,
        ]
    )

    preprocessor = TextPreprocessor(pipeline)

    record = TextRecord(
        text="  HELLO WORLD  ",
        label=2,
    )

    result = preprocessor.process(record)

    assert result.text == "hello world"
    assert result.label == 2


def test_process_preserves_original_record():
    pipeline = TextCleaningPipeline(
        transforms=[
            str.strip,
            lowercase_text,
        ]
    )

    preprocessor = TextPreprocessor(pipeline)

    record = TextRecord(
        text="  HELLO WORLD  ",
        label=1,
    )

    result = preprocessor.process(record)

    assert record.text == "  HELLO WORLD  "
    assert record.label == 1

    assert result is not record


def test_process_many():
    pipeline = TextCleaningPipeline(
        transforms=[
            str.strip,
            lowercase_text,
        ]
    )

    preprocessor = TextPreprocessor(pipeline)

    records = [
        TextRecord(text=" FIRST ARTICLE ", label=0),
        TextRecord(text=" SECOND ARTICLE ", label=1),
        TextRecord(text=" THIRD ARTICLE ", label=2),
    ]

    result = preprocessor.process_many(records)

    assert result == [
        TextRecord(text="first article", label=0),
        TextRecord(text="second article", label=1),
        TextRecord(text="third article", label=2),
    ]


def test_process_empty_collection():
    pipeline = TextCleaningPipeline()

    preprocessor = TextPreprocessor(pipeline)

    assert preprocessor.process_many([]) == []