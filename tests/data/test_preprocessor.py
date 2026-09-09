from nlp_platform.data.preprocessing_config import PreprocessingConfig
from nlp_platform.data.preprocessor import TextPreprocessor
from nlp_platform.data.records import TextRecord


def test_default_configuration():
    preprocessor = TextPreprocessor()

    record = TextRecord(
        text="  HELLO    WORLD  ",
        label=2,
    )

    result = preprocessor.process(record)

    assert result.text == "hello world"
    assert result.label == 2


def test_lowercase_can_be_disabled():
    config = PreprocessingConfig(
        lowercase=False,
    )

    preprocessor = TextPreprocessor(config)

    record = TextRecord(
        text="  HELLO WORLD  ",
        label=1,
    )

    result = preprocessor.process(record)

    assert result.text == "HELLO WORLD"


def test_whitespace_normalization_can_be_disabled():
    config = PreprocessingConfig(
        normalize_whitespace=False,
    )

    preprocessor = TextPreprocessor(config)

    record = TextRecord(
        text="HELLO    WORLD",
        label=1,
    )

    result = preprocessor.process(record)

    assert result.text == "hello    world"


def test_stripping_can_be_disabled():
    config = PreprocessingConfig(
        strip_whitespace=False,
    )

    preprocessor = TextPreprocessor(config)

    record = TextRecord(
        text="  HELLO WORLD  ",
        label=1,
    )

    result = preprocessor.process(record)

    assert result.text == "hello world"


def test_all_transformations_can_be_disabled():
    config = PreprocessingConfig(
        lowercase=False,
        strip_whitespace=False,
        normalize_whitespace=False,
    )

    preprocessor = TextPreprocessor(config)

    record = TextRecord(
        text="  HELLO    WORLD  ",
        label=1,
    )

    result = preprocessor.process(record)

    assert result.text == "  HELLO    WORLD  "


def test_process_many():
    preprocessor = TextPreprocessor()

    records = [
        TextRecord(text=" FIRST ARTICLE ", label=0),
        TextRecord(text=" SECOND ARTICLE ", label=1),
    ]

    result = preprocessor.process_many(records)

    assert result == [
        TextRecord(text="first article", label=0),
        TextRecord(text="second article", label=1),
    ]