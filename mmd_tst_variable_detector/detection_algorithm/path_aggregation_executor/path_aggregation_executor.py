import time
import copy
import logging
import typing as ty
from distributed import Client

from ...mmd_estimator.mmd_estimator import BaseMmdEstimator
from ...datasets.base import BaseDataset
from ...utils.post_process_logger import PostProcessLoggerHandler
from ..base import BaseVariableDetector
from ..pytorch_lightning_trainer import PytorchLightningDefaultArguments
from ..commons import InterpretableMmdTrainParameters
from ...accelerator_optimizations.factory import create_task_dispatcher
from ...logger_unit import handler

from .commons import (
    PathAggregationAlgorithmParameter,
    PathAggregationTaskRequest,
    PathAggregationTaskResult,
    PathAggregationDetectionResult,
)
from .worker import execute_optimization_task_path_aggregation
from .aggregator import PathAggregator

logger = logging.getLogger(f"{__package__}.{__name__}")
logger.addHandler(handler)


class PathAggregationVariableDetector(BaseVariableDetector):
    """MMD-based variable detector implementing regularized test-power path aggregation.

    Inherits from BaseVariableDetector and leverages BaseTaskDispatcher from
    accelerator_optimizations to support single CPU/GPU as well as concurrent
    Dask CPU and concurrent multi-slot GPU execution.
    """

    def __init__(
        self,
        estimator: BaseMmdEstimator,
        regularization_grid: ty.Sequence[float],
        pytorch_trainer_config: ty.Optional[PytorchLightningDefaultArguments] = None,
        base_training_parameter: ty.Optional[InterpretableMmdTrainParameters] = None,
        threshold: float = 0.01,
        selection_strategy: ty.Literal["threshold", "hist_based", "normalized_threshold"] = "threshold",
        weight_transformation: ty.Literal["identity", "bounded"] = "identity",
        path_weights: ty.Optional[ty.Sequence[float]] = None,
        subsampling_splits: int = 1,
        subsampling_ratio: float = 0.8,
        random_seed: ty.Optional[int] = 42,
        save_split_weights: bool = True,
        # Dispatcher & accelerator parameters
        train_accelerator: str = "cpu",
        distributed_mode: str = "single",
        dask_client: ty.Optional[Client] = None,
        dask_scheduler_address: ty.Optional[str] = None,
        distributed_batch_size: int = 1,
        device_id: int = 0,
        resume_checkpoint_saver: ty.Optional[ty.Any] = None,
        post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
        cv_experiment_name: ty.Optional[str] = None,
        worker_fn: ty.Optional[ty.Callable] = None,
        **kwargs: ty.Any,
    ) -> None:
        """Initialize the PathAggregationVariableDetector.

        Parameters
        ----------
        estimator : BaseMmdEstimator
            MMD estimator configured with an ARD product kernel.
        regularization_grid : Sequence[float]
            Sequence of L1 penalty regularization parameters.
        pytorch_trainer_config : Optional[PytorchLightningDefaultArguments]
            Configuration for the PyTorch / Lightning trainer.
        base_training_parameter : Optional[InterpretableMmdTrainParameters]
            Base training parameters for the MMD detector.
        threshold : float
            Cutoff threshold tau > 0 for aggregated importance scores.
        weight_transformation : Literal["identity", "bounded"]
            Transformation rho applied to weights (default: "identity").
        path_weights : Optional[Sequence[float]]
            Path weights w_lambda summing to 1.0 (default: uniform).
        subsampling_splits : int
            Number of subsampling splits B (default: 1, full data).
        subsampling_ratio : float
            Fraction of data to draw per subsample split (default: 0.8).
        random_seed : Optional[int]
            Random seed for subsampling reproducibility.
        train_accelerator : str
            Target compute accelerator: 'cpu', 'gpu', 'cuda', or 'auto'.
        distributed_mode : str
            Execution mode: 'single' or 'dask'.
        dask_client : Optional[Client]
            Active Dask client if distributed_mode is 'dask'.
        dask_scheduler_address : Optional[str]
            Address of Dask scheduler if client is not provided directly.
        distributed_batch_size : int
            Batch size for dispatching tasks.
        device_id : int
            Device index for single-GPU execution.
        resume_checkpoint_saver : Optional[Any]
            Handler for saving intermediate task checkpoints.
        post_process_handler : Optional[PostProcessLoggerHandler]
            Logging handler for metrics and artifacts.
        cv_experiment_name : Optional[str]
            Experiment identifier for tracking and logging.
        worker_fn : Optional[Callable]
            Worker routine executed for each optimization task.
        """
        trainer_config = (
            pytorch_trainer_config
            if pytorch_trainer_config is not None
            else PytorchLightningDefaultArguments()
        )
        super().__init__(
            estimator=estimator,
            pytorch_trainer_config=trainer_config,
            post_process_handler=post_process_handler,
            dask_client=dask_client,
            **kwargs,
        )

        # 1. Path aggregation algorithm parameters
        self.algorithm_parameters = PathAggregationAlgorithmParameter(
            regularization_grid=sorted(list(regularization_grid)),
            threshold=threshold,
            selection_strategy=selection_strategy,
            weight_transformation=weight_transformation,
            path_weights=list(path_weights) if path_weights is not None else None,
            subsampling_splits=subsampling_splits,
            subsampling_ratio=subsampling_ratio,
            random_seed=random_seed,
            save_split_weights=save_split_weights,
        )
        self.base_training_parameter = (
            base_training_parameter
            if base_training_parameter is not None
            else InterpretableMmdTrainParameters()
        )

        # 2. Dispatcher configuration
        self.train_accelerator = train_accelerator
        self.distributed_mode = distributed_mode
        self.dask_scheduler_address = dask_scheduler_address
        self.distributed_batch_size = max(1, distributed_batch_size)
        self.device_id = device_id
        self.resume_checkpoint_saver = resume_checkpoint_saver
        self.cv_experiment_name = cv_experiment_name
        self.worker_fn = (
            worker_fn
            if worker_fn is not None
            else execute_optimization_task_path_aggregation
        )

        # 3. Instantiate task dispatcher via factory
        self.task_dispatcher = create_task_dispatcher(
            train_accelerator=self.train_accelerator,
            distributed_mode=self.distributed_mode,
            dask_client=self.dask_client,
            dask_scheduler_address=self.dask_scheduler_address,
            batch_size=self.distributed_batch_size,
            resume_checkpoint_saver=self.resume_checkpoint_saver,
            post_process_handler=self.post_process_handler,
            cv_experiment_name=self.cv_experiment_name,
            device_id=self.device_id,
            worker_fn=self.worker_fn,
            **kwargs,
        )
    # end def

    def run_detection(
        self,
        training_dataset: BaseDataset,
        validation_dataset: ty.Optional[BaseDataset] = None,
        **kwargs: ty.Any,
    ) -> PathAggregationDetectionResult:
        """Execute the path_aggregation variable detection workflow.

        Dispatches regularized optimization tasks along the regularization path Lambda
        (concurrently if accelerator/dask resources are available), aggregates the
        resulting ARD weights, and selects variables exceeding threshold tau.

        Parameters
        ----------
        training_dataset : BaseDataset
            Training dataset containing sample groups X and Y.
        validation_dataset : Optional[BaseDataset]
            Optional development / validation dataset.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        PathAggregationDetectionResult
            Container with selected variables, aggregated scores, and path weights.
        """
        start_wall_time = time.time()
        dimension_size = self.estimator.kernel_obj.ard_weights.shape[0]

        # 1. Generate task request payloads along the regularization path
        seq_task_requests = self._generate_task_requests(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
        )

        # 2. Execute tasks using the configured task dispatcher (concurrent if mode allows)
        seq_task_results: ty.List[PathAggregationTaskResult] = self.task_dispatcher.dispatch(
            seq_task_requests
        )

        # 3. Aggregate importance scores over the regularization path
        aggregator = PathAggregator(parameters=self.algorithm_parameters)
        aggregated_path_scores = aggregator.aggregate_path_weights(
            task_results=seq_task_results,
            dimension_size=dimension_size,
        )

        # 4. Filter selected variables by threshold
        selected_variables = aggregator.select_variables(
            aggregated_scores=aggregated_path_scores.aggregated_scores
        )

        total_execution_seconds = time.time() - start_wall_time
        num_successful_tasks = sum(1 for res in seq_task_results if res.is_success)

        metadata = {
            "execution_time_total_seconds": total_execution_seconds,
            "num_tasks_total": len(seq_task_requests),
            "num_tasks_successful": num_successful_tasks,
            "train_accelerator": self.train_accelerator,
            "distributed_mode": self.distributed_mode,
            "threshold": self.algorithm_parameters.threshold,
            "selection_strategy": self.algorithm_parameters.selection_strategy,
            "weight_transformation": self.algorithm_parameters.weight_transformation,
            "save_split_weights": self.algorithm_parameters.save_split_weights,
        }

        return PathAggregationDetectionResult(
            selected_variables=selected_variables,
            aggregated_scores=aggregated_path_scores.aggregated_scores,
            regularization_path_weights=aggregated_path_scores.regularization_path_weights,
            split_path_weights=aggregated_path_scores.split_path_weights,
            regularization_grid=self.algorithm_parameters.regularization_grid,
            execution_metadata=metadata,
        )
    # end def

    def _generate_task_requests(
        self,
        training_dataset: BaseDataset,
        validation_dataset: ty.Optional[BaseDataset] = None,
    ) -> ty.List[PathAggregationTaskRequest]:
        """Generate optimization task request payloads across grid Lambda and subsample splits.

        Parameters
        ----------
        training_dataset : BaseDataset
            The source training dataset.
        validation_dataset : Optional[BaseDataset]
            The source validation dataset.

        Returns
        -------
        List[PathAggregationTaskRequest]
            List of individual task requests ready for dispatcher execution.
        """
        grid = self.algorithm_parameters.regularization_grid
        splits_count = self.algorithm_parameters.subsampling_splits
        ratio = self.algorithm_parameters.subsampling_ratio
        root_seed = self.algorithm_parameters.random_seed or 42

        seq_requests: ty.List[PathAggregationTaskRequest] = []

        for split_idx in range(splits_count):
            # If subsampling splits > 1, draw subsampled training dataset
            if splits_count > 1:
                split_seed = root_seed + split_idx
                n_samples_sub = max(1, int(len(training_dataset) * ratio))
                _, dataset_train_split = training_dataset.get_subsample_dataset(
                    n_samples=n_samples_sub,
                    random_seed=split_seed,
                )
            else:
                dataset_train_split = training_dataset
            # end if

            for lambda_val in grid:
                task_id = f"lambda_{lambda_val:e}-split_{split_idx}"
                request = PathAggregationTaskRequest(
                    task_id=task_id,
                    lambda_val=lambda_val,
                    split_index=split_idx,
                    estimator=self.estimator,
                    dataset_train=dataset_train_split,
                    dataset_validation=validation_dataset,
                    training_parameter=self.base_training_parameter,
                    pytorch_trainer_config=self.pytorch_trainer_config,
                )
                seq_requests.append(request)
            # end for
        # end for

        return seq_requests
    # end def
# end class


# Alias for compatibility with implementation plan
PathAggregationExecutor = PathAggregationVariableDetector
