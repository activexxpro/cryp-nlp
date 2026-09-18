import random

import numpy as np
import pytest
import torch

from nlp_platform.ml.seed import set_seed


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError):
        set_seed(-1)


def test_python_random_is_reproducible() -> None:
    set_seed(42)
    first = [random.random() for _ in range(5)]

    set_seed(42)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_numpy_random_is_reproducible() -> None:
    set_seed(42)
    first = np.random.random(5)

    set_seed(42)
    second = np.random.random(5)

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_torch_random_is_reproducible() -> None:
    set_seed(42)
    first = torch.rand(5)

    set_seed(42)
    second = torch.rand(5)

    assert torch.equal(
        first,
        second,
    )


def test_different_seeds_produce_different_python_values() -> None:
    set_seed(42)
    first = random.random()

    set_seed(123)
    second = random.random()

    assert first != second


def test_different_seeds_produce_different_torch_values() -> None:
    set_seed(42)
    first = torch.rand(5)

    set_seed(123)
    second = torch.rand(5)

    assert not torch.equal(
        first,
        second,
    )