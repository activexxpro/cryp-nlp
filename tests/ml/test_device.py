import pytest
import torch

from nlp_platform.ml.device import DeviceConfig, get_device


def test_default_device_config_uses_auto() -> None:
    config = DeviceConfig()

    assert config.device == "auto"


def test_custom_cpu_device_config() -> None:
    config = DeviceConfig(device="cpu")

    assert config.device == "cpu"


def test_invalid_device_config_raises() -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        DeviceConfig(device="tpu")


def test_cpu_device_selection() -> None:
    device = get_device(DeviceConfig(device="cpu"))

    assert device == torch.device("cpu")


def test_auto_device_selection() -> None:
    device = get_device()

    assert device.type in {"cpu", "cuda"}


def test_cuda_selection_matches_availability() -> None:
    if torch.cuda.is_available():
        device = get_device(DeviceConfig(device="cuda"))

        assert device.type == "cuda"
    else:
        with pytest.raises(
            RuntimeError,
            match="CUDA was requested but is not available",
        ):
            get_device(DeviceConfig(device="cuda"))