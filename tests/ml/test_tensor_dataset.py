import torch

from nlp_platform.data.records import TextRecord
from nlp_platform.nlp.batch_encoder import BatchEncoder
from nlp_platform.nlp.dataset_encoder import DatasetEncoder
from nlp_platform.nlp.encoded_dataset import DatasetEncodingService
from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.nlp_pipeline import NLPPreprocessingPipeline
from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.tokenizer import BasicTokenizer
from nlp_platform.nlp.vocabulary import Vocabulary
from nlp_platform.ml.tensor_dataset import (
    TensorDataset,
    to_tensor_dataset,
)


def create_encoded_dataset():
    vocabulary = Vocabulary.build(
        [
            ["hello", "world"],
            ["hello", "nlp"],
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
        ]
    )


def test_to_tensor_dataset() -> None:
    encoded_dataset = create_encoded_dataset()

    result = to_tensor_dataset(encoded_dataset)

    assert isinstance(result, TensorDataset)
    assert isinstance(result.input_ids, torch.Tensor)
    assert isinstance(result.labels, torch.Tensor)


def test_input_ids_have_long_dtype() -> None:
    dataset = create_encoded_dataset()

    result = to_tensor_dataset(dataset)

    assert result.input_ids.dtype == torch.long


def test_labels_have_long_dtype() -> None:
    dataset = create_encoded_dataset()

    result = to_tensor_dataset(dataset)

    assert result.labels.dtype == torch.long


def test_tensor_shapes() -> None:
    dataset = create_encoded_dataset()

    result = to_tensor_dataset(dataset)

    assert result.input_ids.shape == (2, 4)
    assert result.labels.shape == (2,)


def test_tensor_values_are_preserved() -> None:
    dataset = create_encoded_dataset()

    result = to_tensor_dataset(dataset)

    assert result.input_ids.tolist() == [
        [2, 4, 0, 0],
        [2, 3, 0, 0],
    ]

    assert result.labels.tolist() == [0, 1]


def test_dataset_size() -> None:
    dataset = create_encoded_dataset()

    result = to_tensor_dataset(dataset)

    assert result.size == 2


def test_sequence_length() -> None:
    dataset = create_encoded_dataset()

    result = to_tensor_dataset(dataset)

    assert result.sequence_length == 4


def test_empty_dataset() -> None:
    from nlp_platform.nlp.encoded_dataset import EncodedDataset

    dataset = EncodedDataset(inputs=())

    result = to_tensor_dataset(dataset)

    assert result.size == 0
    assert result.input_ids.shape == (0,)
    assert result.labels.shape == (0,)