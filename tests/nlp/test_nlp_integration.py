from nlp_platform.nlp.batch_encoder import BatchEncoder
from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.nlp_pipeline import NLPPreprocessingPipeline
from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.tokenizer import BasicTokenizer
from nlp_platform.nlp.vocabulary import Vocabulary


def create_components() -> NLPPreprocessingPipeline:
    tokenizer = BasicTokenizer()

    vocabulary = Vocabulary.build(
        [
            ["hello", "world"],
            ["hello", "nlp"],
            ["machine", "learning"],
        ]
    )

    encoder = TokenEncoder(vocabulary)

    padder = SequencePadder(
        vocabulary=vocabulary,
        max_length=5,
    )

    batch_encoder = BatchEncoder(
        tokenizer=tokenizer,
        encoder=encoder,
        padder=padder,
    )

    return NLPPreprocessingPipeline(
        batch_encoder=batch_encoder,
    )


def test_full_pipeline_processes_text() -> None:
    pipeline = create_components()

    result = pipeline.process(
        "Hello world"
    )

    assert result == [2, 6, 0, 0, 0]


def test_full_pipeline_handles_case_normalization() -> None:
    pipeline = create_components()

    result = pipeline.process(
        "HELLO WORLD"
    )

    assert result == [2, 6, 0, 0, 0]


def test_full_pipeline_handles_unknown_words() -> None:
    pipeline = create_components()

    result = pipeline.process(
        "hello completely_unknown"
    )

    assert result == [2, 1, 0, 0, 0]


def test_full_pipeline_truncates_long_sequences() -> None:
    pipeline = create_components()

    result = pipeline.process(
        "hello world machine learning hello world"
    )

    assert len(result) == 5
    assert result == [2, 6, 4, 3, 2]


def test_full_pipeline_processes_batch() -> None:
    pipeline = create_components()

    result = pipeline.process_many(
        [
            "hello",
            "hello world",
            "hello nlp",
        ]
    )

    assert result == [
        [2, 0, 0, 0, 0],
        [2, 6, 0, 0, 0],
        [2, 5, 0, 0, 0],
    ]


def test_full_pipeline_preserves_batch_order() -> None:
    pipeline = create_components()

    texts = [
        "machine learning",
        "hello world",
        "hello nlp",
    ]

    result = pipeline.process_many(texts)

    assert result == [
        [4, 3, 0, 0, 0],
        [2, 6, 0, 0, 0],
        [2, 5, 0, 0, 0],
    ]


def test_vocabulary_round_trip() -> None:
    vocabulary = Vocabulary.build(
        [
            ["hello", "world"],
            ["hello", "nlp"],
        ]
    )

    encoder = TokenEncoder(vocabulary)

    tokens = ["hello", "world", "nlp"]

    encoded = encoder.encode(tokens)
    decoded = encoder.decode(encoded)

    assert decoded == tokens


def test_padding_preserves_encoded_tokens() -> None:
    vocabulary = Vocabulary.build(
        [
            ["hello", "world"],
        ]
    )

    padder = SequencePadder(
        vocabulary=vocabulary,
        max_length=4,
    )

    result = padder.pad_or_truncate([2, 3])

    assert result[:2] == [2, 3]
    assert result[2:] == [0, 0]


def test_unknown_tokens_use_unk_id() -> None:
    vocabulary = Vocabulary.build(
        [
            ["hello", "world"],
        ]
    )

    encoder = TokenEncoder(vocabulary)

    result = encoder.encode(
        ["hello", "missing"]
    )

    assert result == [2, 1]