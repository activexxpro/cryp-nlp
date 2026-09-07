from nlp_platform.nlp.batch_encoder import BatchEncoder
from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.nlp_pipeline import NLPPreprocessingPipeline
from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.tokenizer import BasicTokenizer
from nlp_platform.nlp.vocabulary import Vocabulary


def create_pipeline() -> NLPPreprocessingPipeline:
    vocabulary = Vocabulary.build(
        [
            ["hello", "world"],
            ["hello", "nlp"],
            ["machine", "learning"],
        ]
    )

    tokenizer = BasicTokenizer()

    encoder = TokenEncoder(vocabulary)

    padder = SequencePadder(
        vocabulary=vocabulary,
        max_length=4,
    )

    batch_encoder = BatchEncoder(
        tokenizer=tokenizer,
        encoder=encoder,
        padder=padder,
    )

    return NLPPreprocessingPipeline(
        batch_encoder=batch_encoder,
    )


def test_process_text() -> None:
    pipeline = create_pipeline()

    result = pipeline.process("Hello world")

    assert result == [2, 6, 0, 0]


def test_process_lowercases_text() -> None:
    pipeline = create_pipeline()

    result = pipeline.process("HELLO WORLD")

    assert result == [2, 6, 0, 0]


def test_process_unknown_tokens() -> None:
    pipeline = create_pipeline()

    result = pipeline.process("hello unknown")

    assert result == [2, 1, 0, 0]


def test_process_empty_text() -> None:
    pipeline = create_pipeline()

    result = pipeline.process("")

    assert result == [0, 0, 0, 0]


def test_process_many() -> None:
    pipeline = create_pipeline()

    result = pipeline.process_many(
        [
            "hello world",
            "hello nlp",
        ]
    )

    assert result == [
        [2, 6, 0, 0],
        [2, 5, 0, 0],
    ]


def test_process_many_empty_input() -> None:
    pipeline = create_pipeline()

    assert pipeline.process_many([]) == []


def test_process_rejects_non_string() -> None:
    pipeline = create_pipeline()

    try:
        pipeline.process(123)  # type: ignore[arg-type]
    except TypeError as exc:
        assert str(exc) == "text must be a string"
    else:
        raise AssertionError("expected TypeError")