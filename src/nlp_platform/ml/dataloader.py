from torch.utils.data import DataLoader

from nlp_platform.ml.dataloader_config import DataLoaderConfig
from nlp_platform.ml.torch_dataset import NLPDataset


def create_dataloader(
    dataset: NLPDataset,
    config: DataLoaderConfig | None = None,
) -> DataLoader:
    """Create a PyTorch DataLoader for an NLP dataset."""

    loader_config = config or DataLoaderConfig()

    return DataLoader(
        dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        num_workers=loader_config.num_workers,
        drop_last=loader_config.drop_last,
    )