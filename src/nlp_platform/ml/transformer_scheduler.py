from dataclasses import dataclass

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for transformer learning-rate scheduling."""

    total_steps: int
    warmup_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise ValueError("total_steps must be at least 1")

        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be between 0 and 1")

    @property
    def warmup_steps(self) -> int:
        """Return the number of warmup steps."""
        return int(self.total_steps * self.warmup_ratio)


def create_transformer_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
) -> LambdaLR:
    """Create a linear warmup and decay scheduler."""
    warmup_steps = config.warmup_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            if warmup_steps == 0:
                return 1.0
            return float(current_step) / float(warmup_steps)

        remaining_steps = config.total_steps - current_step
        decay_steps = max(config.total_steps - warmup_steps, 1)

        return max(
            0.0,
            float(remaining_steps) / float(decay_steps),
        )

    return LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )