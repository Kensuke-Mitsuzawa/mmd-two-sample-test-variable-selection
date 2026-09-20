from abc import ABC, abstractmethod
import typing as ty
from distributed import Client

from ..mmd_estimator.mmd_estimator import BaseMmdEstimator
from ..datasets.base import BaseDataset
from ..utils.post_process_logger import PostProcessLoggerHandler
from .pytorch_lightning_trainer import PytorchLightningDefaultArguments


class BaseVariableDetector(ABC):
    """Abstract base executor for MMD-based variable selection algorithms.

    Provides the common execution contract and shared components across
    Algorithm One, Baseline MMD, and Cross-Validation MMD approaches.

    Parameters
    ----------
    estimator : BaseMmdEstimator
        MMD estimator instance configured with kernel and ARD weights.
    pytorch_trainer_config : PytorchLightningDefaultArguments
        Configuration for the PyTorch Lightning trainer.
    post_process_handler : Optional[PostProcessLoggerHandler]
        Handler for metrics logging and visualization exports.
    dask_client : Optional[Client]
        Dask client for distributed or concurrent GPU execution.
    """

    def __init__(
        self,
        estimator: BaseMmdEstimator,
        pytorch_trainer_config: PytorchLightningDefaultArguments,
        post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
        dask_client: ty.Optional[Client] = None,
        **kwargs: ty.Any,
    ) -> None:
        self.estimator = estimator
        self.pytorch_trainer_config = pytorch_trainer_config
        self.post_process_handler = post_process_handler
        self.dask_client = dask_client
        self.extra_kwargs = kwargs
    # end def

    @abstractmethod
    def run_detection(
        self,
        training_dataset: BaseDataset,
        validation_dataset: ty.Optional[BaseDataset] = None,
        **kwargs: ty.Any,
    ) -> ty.Any:
        """Execute the variable detection algorithm.

        Parameters
        ----------
        training_dataset : BaseDataset
            Training dataset containing sample groups X and Y.
        validation_dataset : Optional[BaseDataset]
            Optional development or validation dataset.
        **kwargs : Any
            Additional algorithm-specific execution arguments.

        Returns
        -------
        Any
            Algorithm-specific detection result container.
        """
        pass
    # end def
# end class


# Alias for backward compatibility and conceptual clarity
BaseMmdOptimizationExecutor = BaseVariableDetector
