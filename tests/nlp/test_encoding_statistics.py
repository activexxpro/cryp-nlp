from nlp_platform.data.records import TextRecord
from nlp_platform.nlp.batch_encoder import BatchEncoder
from nlp_platform.nlp.dataset_encoder import DatasetEncoder
from nlp_platform.nlp.encoded_dataset import (
    DatasetEncodingService,
    EncodedDataset,
)
from nlp_platform.nlp.encoding_statistics import (
    calculate_encoding_statistics,
)
from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.nlp_pipeline import NLPPreprocessingPipeline
from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.tokenizer import BasicTokenizer
from nlp_platform.nlp.vocabulary import Vocabulary


def create_dataset() -> EncodedDataset:
    vocabulary = Vocabulary.build(
        [
            ["hello", "world"],
            ["hello", "nlp"],
            ["machine", "learning"],
        ]
    )

    tokenizer = BasicTokenizer()
    token_encoder = TokenEncoder(vocabulary)

    padder = SequencePadder(
        vocabulary=vocabulary,
        max_length=4,
    )

    batch_encoder = BatchEncoder(
        tokenizer=tokenizer,
        encoder=token_encoder,
        padder=padder,
    )

    pipeline = NLPPreprocessingPipeline(
        batch_encoder=batch_encoder,
    )

    dataset_encoder = DatasetEncoder(
        pipeline=pipeline,
    )

    service = DatasetEncodingService(
        encoder=dataset_encoder,
    )

    return service.encode(
        [
            TextRecord("hello world", 0),
            TextRecord("hello nlp", 1),
            TextRecord("machine learning", 1),
        ]
    )


def test_calculate_statistics() -> None:
    dataset = create_dataset()

    stats = calculate_encoding_statistics(dataset)

    assert stats.total_samples == 3
    assert stats.sequence_length == 4
    assert stats.num_classes == 2
    assert stats.label_counts == {
        0: 1,
        1: 2,
    }


def test_majority_class() -> None:
    dataset = create_dataset()

    stats = calculate_encoding_statistics(dataset)

    assert stats.majority_class == 1


def test_minority_class() -> None:
    dataset = create_dataset()

    stats = calculate_encoding_statistics(dataset)

    assert stats.minority_class == 0


def test_empty_dataset_statistics() -> None:
    dataset = EncodedDataset(inputs=())

    stats = calculate_encoding_statistics(dataset)

    assert stats.total_samples == 0
    assert stats.sequence_length == 0
    assert stats.num_classes == 0
    assert stats.label_counts == {}
    assert stats.majority_class is None
    assert stats.minority_class is None


def test_label_counts_are_sorted() -> None:
    dataset = create_dataset()

    stats = calculate_encoding_statistics(dataset)

    assert list(stats.label_counts.keys()) == [0, 1]


def test_inconsistent_sequence_lengths_raise_error() -> None:
    from nlp_platform.nlp.model_input import ModelInput

    dataset = EncodedDataset(
        inputs=(
            ModelInput(
                input_ids=(1, 2, 3),
                label=0,
            ),
            ModelInput(
                input_ids=(1, 2),
                label=1,
            ),
        )
    )

    try:
        calculate_encoding_statistics(dataset)
    except ValueError as exc:
        assert (
            str(exc)
            == "all encoded sequences must have the same length"
        )
    else:
        raise AssertionError("expected ValueError")