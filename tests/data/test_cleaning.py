from nlp_platform.data.cleaning import (
    TextCleaningPipeline,
    lowercase_text,
    remove_extra_punctuation,
)


def test_default_pipeline_normalizes_whitespace():
    pipeline = TextCleaningPipeline()

    result = pipeline.transform("  Hello    world  ")

    assert result == "Hello world"


def test_custom_pipeline_applies_transforms_in_order():
    pipeline = TextCleaningPipeline(
        transforms=[
            str.strip,
            lowercase_text,
        ]
    )

    result = pipeline.transform("  Hello WORLD  ")

    assert result == "hello world"


def test_lowercase_transform():
    assert lowercase_text("Hello WORLD") == "hello world"


def test_repeated_exclamation_marks_are_normalized():
    result = remove_extra_punctuation("Great!!!!")

    assert result == "Great!"


def test_repeated_question_marks_are_normalized():
    result = remove_extra_punctuation("Really????")

    assert result == "Really?"


def test_mixed_transforms():
    pipeline = TextCleaningPipeline(
        transforms=[
            str.strip,
            lowercase_text,
            remove_extra_punctuation,
        ]
    )

    result = pipeline.transform(
        "  AMAZING!!!! NEWS????  "
    )

    assert result == "amazing! news?"


def test_transform_many():
    pipeline = TextCleaningPipeline(
        transforms=[
            str.strip,
            lowercase_text,
        ]
    )

    result = pipeline.transform_many(
        [
            "  HELLO  ",
            " WORLD ",
            " NLP ",
        ]
    )

    assert result == [
        "hello",
        "world",
        "nlp",
    ]


def test_empty_pipeline():
    pipeline = TextCleaningPipeline(transforms=[])

    result = pipeline.transform("  Hello   WORLD  ")

    assert result == "  Hello   WORLD  "