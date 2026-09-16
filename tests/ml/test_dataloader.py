from nlp_platform.data.records import TextRecord
from nlp_platform.ml.dataloader import create_dataloader
from nlp_platform.ml.dataloader_config import DataLoaderConfig
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


def create_dataset() -> NLPDataset:
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
            TextRecord("hello world", 0),
            TextRecord("hello nlp", 1),
            TextRecord("hello world", 0),
        ]
    )

    tensors = to_tensor_dataset(encoded)

    return NLPDataset(tensors)


def test_dataloader_batch_size() -> None:
    dataset = create_dataset()

    loader = create_dataloader(
        dataset,
        DataLoaderConfig(
            batch_size=2,
            shuffle=False,
        ),
    )

    batches = list(loader)

    assert len(batches) == 3


def test_batch_shapes() -> None:
    dataset = create_dataset()

    loader = create_dataloader(
        dataset,
        DataLoaderConfig(
            batch_size=2,
            shuffle=False,
        ),
    )

    batch = next(iter(loader))

    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].shape == (2,)


def test_batch_values() -> None:
    dataset = create_dataset()

    loader = create_dataloader(
        dataset,
        DataLoaderConfig(
            batch_size=2,
            shuffle=False,
        ),
    )

    batch = next(iter(loader))

    assert batch["input_ids"].tolist() == [
        [2, 4, 0, 0],
        [2, 3, 0, 0],
    ]

    assert batch["labels"].tolist() == [0, 1]


def test_last_partial_batch_is_preserved() -> None:
    dataset = create_dataset()

    loader = create_dataloader(
        dataset,
        DataLoaderConfig(
            batch_size=2,
            shuffle=False,
            drop_last=False,
        ),
    )

    batches = list(loader)

    assert batches[-1]["input_ids"].shape == (1, 4)
    assert batches[-1]["labels"].shape == (1,)


def test_drop_last_removes_partial_batch() -> None:
    dataset = create_dataset()

    loader = create_dataloader(
        dataset,
        DataLoaderConfig(
            batch_size=2,
            shuffle=False,
            drop_last=True,
        ),
    )

    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert batch["input_ids"].shape == (2, 4)
        assert batch["labels"].shape == (2,)


def test_shuffle_can_be_disabled() -> None:
    dataset = create_dataset()

    loader = create_dataloader(
        dataset,
        DataLoaderConfig(
            batch_size=5,
            shuffle=False,
        ),
    )

    batch = next(iter(loader))

    assert batch["labels"].tolist() == [
        0,
        1,
        0,
        1,
        0,
    ]


def test_default_configuration_creates_loader() -> None:
    dataset = create_dataset()

    loader = create_dataloader(dataset)

    assert loader.batch_size == 32