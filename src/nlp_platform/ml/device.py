from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceConfig:
    """Configuration for selecting the PyTorch device."""

    device: str = "auto"

    def __post_init__(self) -> None:
        allowed_devices = {"auto", "cpu", "cuda"}

        if self.device not in allowed_devices:
            raise ValueError(
                f"device must be one of {sorted(allowed_devices)}"
            )


def get_device(
    config: DeviceConfig | None = None,
) -> torch.device:
    """Select the PyTorch device according to configuration."""

    device_config = config or DeviceConfig()

    if device_config.device == "cpu":
        return torch.device("cpu")

    if device_config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available"
            )

        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")