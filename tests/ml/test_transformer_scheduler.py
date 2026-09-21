import pytest
import torch

from nlp_platform.ml.transformer_scheduler import (
    SchedulerConfig,
    create_transformer_scheduler,
)


def create_optimizer() -> torch.optim.Optimizer:
    model = torch.nn.Linear(4, 2)

    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )


def test_scheduler_config_defaults() -> None:
    config = SchedulerConfig(total_steps=100)

    assert config.total_steps == 100
    assert config.warmup_ratio == 0.1
    assert config.warmup_steps == 10


def test_scheduler_config_custom_warmup() -> None:
    config = SchedulerConfig(
        total_steps=200,
        warmup_ratio=0.25,
    )

    assert config.warmup_steps == 50


@pytest.mark.parametrize(
    "total_steps",
    [0, -1],
)
def test_scheduler_rejects_invalid_total_steps(
    total_steps: int,
) -> None:
    with pytest.raises(ValueError, match="total_steps"):
        SchedulerConfig(total_steps=total_steps)


@pytest.mark.parametrize(
    "warmup_ratio",
    [-0.1, 1.1],
)
def test_scheduler_rejects_invalid_warmup_ratio(
    warmup_ratio: float,
) -> None:
    with pytest.raises(ValueError, match="warmup_ratio"):
        SchedulerConfig(
            total_steps=100,
            warmup_ratio=warmup_ratio,
        )


def test_scheduler_can_be_created() -> None:
    optimizer = create_optimizer()

    scheduler = create_transformer_scheduler(
        optimizer,
        SchedulerConfig(
            total_steps=10,
            warmup_ratio=0.2,
        ),
    )

    assert scheduler is not None


def test_scheduler_updates_learning_rate() -> None:
    optimizer = create_optimizer()

    scheduler = create_transformer_scheduler(
        optimizer,
        SchedulerConfig(
            total_steps=10,
            warmup_ratio=0.2,
        ),
    )

    learning_rates = []

    for _ in range(5):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[0] > 0.0
    assert learning_rates[-1] < 1e-3


def test_zero_warmup_keeps_initial_learning_rate_available() -> None:
    optimizer = create_optimizer()

    scheduler = create_transformer_scheduler(
        optimizer,
        SchedulerConfig(
            total_steps=10,
            warmup_ratio=0.0,
        ),
    )

    optimizer.step()
    scheduler.step()

    assert optimizer.param_groups[0]["lr"] > 0.0