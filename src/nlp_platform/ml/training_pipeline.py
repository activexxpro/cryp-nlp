from pathlib import Path

from torch import nn

from nlp_platform.ml.best_checkpoint import BestCheckpointTracker
from nlp_platform.ml.checkpoint import CheckpointManager
from nlp_platform.ml.dataloader import create_dataloader
from nlp_platform.ml.dataloader_config import DataLoaderConfig
from nlp_platform.ml.device import DeviceConfig, get_device
from nlp_platform.ml.early_stopping import (
    EarlyStopping,
    EarlyStoppingConfig,
)
from nlp_platform.ml.evaluator import Evaluator
from nlp_platform.ml.optimizer import create_optimizer
from nlp_platform.ml.tensor_dataset import to_tensor_dataset
from nlp_platform.ml.torch_dataset import NLPDataset
from nlp_platform.ml.trainer import Trainer
from nlp_platform.ml.training_history import TrainingHistory
from nlp_platform.ml.training_run_config import TrainingRunConfig
from nlp_platform.ml.training_service import (
    TrainingRunResult,
    TrainingService,
)
from nlp_platform.nlp.training_dataset import TrainingDataset

from nlp_platform.ml.experiment_config import ExperimentConfig
from nlp_platform.ml.seed import set_seed

from nlp_platform.ml.experiment_result import ExperimentResult

class TrainingPipeline:
    """Execute a complete NLP model training pipeline."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingRunConfig | None = None,
        device_config: DeviceConfig | None = None,
        dataloader_config: DataLoaderConfig | None = None,
        checkpoint_directory: str | Path | None = None,
        experiment_config: ExperimentConfig | None = None,
    ) -> None:
        self._model = model
        self._config = config or TrainingRunConfig()
        self._device = get_device(device_config)
        self._dataloader_config = (
            dataloader_config or DataLoaderConfig()
        )
        self._checkpoint_directory = checkpoint_directory
        self._experiment_config = (
            experiment_config or ExperimentConfig()
        )


    def run(
        self,
        dataset: TrainingDataset,
    ) -> ExperimentResult:
        """Train and evaluate the configured model."""

        set_seed(
            self._experiment_config.seed
        )

        train_tensors = to_tensor_dataset(
            dataset.train
        )
        validation_tensors = to_tensor_dataset(
            dataset.validation
        )

        train_loader = create_dataloader(
            NLPDataset(train_tensors),
            self._dataloader_config,
        )

        validation_loader = create_dataloader(
            NLPDataset(validation_tensors),
            DataLoaderConfig(
                batch_size=self._dataloader_config.batch_size,
                shuffle=False,
                num_workers=self._dataloader_config.num_workers,
                drop_last=False,
            ),
        )

        self._model.to(self._device)

        optimizer = create_optimizer(
            self._model,
        )

        # Re-create the optimizer with the configured
        # training hyperparameters.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = self._config.learning_rate
            parameter_group["weight_decay"] = (
                self._config.weight_decay
            )

        trainer = Trainer(
            model=self._model,
            optimizer=optimizer,
            device=self._device,
        )

        evaluator = Evaluator(
            model=self._model,
            device=self._device,
        )

        checkpoint_tracker = None

        if self._checkpoint_directory is not None:
            checkpoint_manager = CheckpointManager(
                self._checkpoint_directory
            )

            checkpoint_tracker = BestCheckpointTracker(
                checkpoint_manager
            )

        early_stopping = None

        if self._config.early_stopping_enabled:
            early_stopping = EarlyStopping(
                EarlyStoppingConfig(
                    patience=self._config.early_stopping_patience,
                    min_delta=self._config.early_stopping_min_delta,
                )
            )

        service = TrainingService(
            trainer=trainer,
            evaluator=evaluator,
            epochs=self._config.epochs,
            checkpoint_tracker=checkpoint_tracker,
            early_stopping=early_stopping,
        )

        training_result = service.train(
            train_dataloader=train_loader,
            validation_dataloader=validation_loader,
        )

        return ExperimentResult(
            experiment_name=self._experiment_config.name,
            seed=self._experiment_config.seed,
            history=training_result.history,
            epochs_completed=training_result.epochs_completed,
            stopped_early=training_result.stopped_early,
            best_epoch=training_result.best_epoch,
            best_validation_loss=(
                training_result.best_validation_loss
            ),
            best_checkpoint=(
                str(checkpoint_tracker.best_path)
                if (
                    checkpoint_tracker is not None
                    and checkpoint_tracker.best_path is not None
                )
                else None
            ),
        )