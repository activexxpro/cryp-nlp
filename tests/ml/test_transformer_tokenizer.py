from typing import Any

from nlp_platform.ml.transformer_config import TransformerConfig
from nlp_platform.ml.transformer_tokenizer import (
    TokenizedBatch,
    TransformerTokenizer,
)


class FakeTokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        padding: str,
        truncation: bool,
        max_length: int,
        return_attention_mask: bool,
    ) -> dict[str, list[list[int]]]:
        assert padding == "max_length"
        assert truncation is True
        assert return_attention_mask is True

        return {
            "input_ids": [
                [101, 10, 11, 0][:max_length],
                [101, 20, 21, 0][:max_length],
            ],
            "attention_mask": [
                [1, 1, 1, 0][:max_length],
                [1, 1, 1, 0][:max_length],
            ],
        }


def test_tokenized_batch_properties() -> None:
    batch = TokenizedBatch(
        input_ids=(
            (101, 10, 11, 0),
            (101, 20, 21, 0),
        ),
        attention_mask=(
            (1, 1, 1, 0),
            (1, 1, 1, 0),
        ),
    )

    assert batch.size == 2
    assert batch.sequence_length == 4


def test_empty_tokenized_batch() -> None:
    batch = TokenizedBatch(
        input_ids=(),
        attention_mask=(),
    )

    assert batch.size == 0
    assert batch.sequence_length == 0


def test_transformer_tokenizer_encodes_batch() -> None:
    tokenizer = TransformerTokenizer(
        tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
        config=TransformerConfig(
            max_length=4,
        ),
    )

    result = tokenizer.encode(
        ["first document", "second document"],
    )

    assert isinstance(result, TokenizedBatch)
    assert result.size == 2
    assert result.sequence_length == 4
    assert result.input_ids == (
        (101, 10, 11, 0),
        (101, 20, 21, 0),
    )
    assert result.attention_mask == (
        (1, 1, 1, 0),
        (1, 1, 1, 0),
    )


def test_transformer_tokenizer_handles_empty_input() -> None:
    tokenizer = TransformerTokenizer(
        tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
    )

    result = tokenizer.encode([])

    assert result.input_ids == ()
    assert result.attention_mask == ()
    assert result.size == 0


def test_model_max_length_comes_from_config() -> None:
    tokenizer = TransformerTokenizer(
        tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
        config=TransformerConfig(
            max_length=256,
        ),
    )

    assert tokenizer.model_max_length == 256