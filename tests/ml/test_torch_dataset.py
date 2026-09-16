import torch

from nlp_platform.data.records import TextRecord
from nlp_platform.ml.tensor_dataset import to_tensor_dataset
from nlp_platform.ml.torch_dataset import NLPDataset
from nlp_platform.nlp.batch_encoder import BatchEncoder
from nlp_platform.nlp.dataset_encoder import DatasetEncoder
from nlp_platform.nlp.encoded_dataset import DatasetEncodingService
from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.nlp_pipeline import NLPPreprocessingPipeline
from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.tokenizer import BasicTokenizer
from nlp_platform.nlp.vocabulary import Vocabulary


def create_torch_dataset() -> NLPDataset:
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

    encoded = service.encode(
        [
            TextRecord("hello world", 0),
            TextRecord("hello nlp", 1),
        ]
    )

    tensors = to_tensor_dataset(encoded)

    return NLPDataset(tensors)


def test_dataset_length() -> None:
    dataset = create_torch_dataset()

    assert len(dataset) == 2


def test_getitem_returns_dictionary() -> None:
    dataset = create_torch_dataset()

    item = dataset[0]

    assert set(item) == {
        "input_ids",
        "labels",
    }


def test_input_ids_are_tensor() -> None:
    dataset = create_torch_dataset()

    item = dataset[0]

    assert isinstance(
        item["input_ids"],
        torch.Tensor,
    )


def test_labels_are_tensor() -> None:
    dataset = create_torch_dataset()

    item = dataset[0]

    assert isinstance(
        item["labels"],
        torch.Tensor,
    )


def test_input_ids_values() -> None:
    dataset = create_torch_dataset()

    item = dataset[0]

    assert item["input_ids"].tolist() == [
        2,
        4,
        0,
        0,
    ]


def test_label_value() -> None:
    dataset = create_torch_dataset()

    first = dataset[0]
    second = dataset[1]

    assert first["labels"].item() == 0
    assert second["labels"].item() == 1


def test_dataset_items_preserve_order() -> None:
    dataset = create_torch_dataset()

    first = dataset[0]
    second = dataset[1]

    assert first["input_ids"].tolist() == [
        2,
        4,
        0,
        0,
    ]

    assert second["input_ids"].tolist() == [
        2,
        3,
        0,
        0,
    ]


def test_dataset_supports_last_index() -> None:
    dataset = create_torch_dataset()

    item = dataset[-1]

    assert item["labels"].item() == 1