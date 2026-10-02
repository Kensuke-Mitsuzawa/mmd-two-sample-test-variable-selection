import time
import copy
import logging
import typing as ty
import torch

from ...detection_algorithm.commons import (
    RegularizationParameter,
    InterpretableMmdTrainParameters,
)
from ...detection_algorithm.pytorch_lightning_trainer import (
    create_mmd_trainer,
    get_mmd_detector_class,
)
from ...logger_unit import handler
from .commons import PathAggregationTaskRequest, PathAggregationTaskResult

logger = logging.getLogger(f"{__package__}.{__name__}")
logger.addHandler(handler)


def execute_optimization_task_path_aggregation(
    request: PathAggregationTaskRequest,
) -> PathAggregationTaskResult:
    """Execute single regularized test-power optimization task for a given lambda.

    Solves:
        max_{a >= 0} { log J_n(a) - lambda * ||a||_1 }

    Parameters
    ----------
    request : PathAggregationTaskRequest
        Task specification including dataset, estimator, lambda, and trainer config.

    Returns
    -------
    PathAggregationTaskResult
        Task execution result containing trained ARD weights and convergence metrics.
    """
    start_wall_time = time.time()
    try:
        # 1. Deepcopy estimator and initialize ARD weights to ones
        estimator_copy = copy.deepcopy(request.estimator)
        initial_weights = torch.ones(estimator_copy.kernel_obj.ard_weights.shape)
        estimator_copy.kernel_obj.ard_weights = torch.nn.Parameter(initial_weights)

        # 2. Materialize in-memory datasets if file-backed
        if request.dataset_train.is_dataset_on_ram():
            dataset_train = request.dataset_train.generate_dataset_on_ram()
        else:
            dataset_train = request.dataset_train
        # end if

        if request.dataset_validation is None:
            dataset_val = dataset_train
        else:
            if request.dataset_validation.is_dataset_on_ram():
                dataset_val = request.dataset_validation.generate_dataset_on_ram()
            else:
                dataset_val = request.dataset_validation
            # end if
        # end if

        # 3. Configure training parameter with pure L1 penalty
        train_params = copy.deepcopy(request.training_parameter)
        train_params.regularization_parameter = RegularizationParameter(
            lambda_1=request.lambda_val,
            lambda_2=0.0,
        )
        train_params.is_use_log = 1
        train_params.objective_function = "ratio"

        # 4. Instantiate detector module and trainer
        use_legacy = getattr(train_params, "use_legacy_optimization", False) or \
                     getattr(request.pytorch_trainer_config, "use_legacy_optimization", False)
        detector_cls = get_mmd_detector_class(use_legacy_optimization=use_legacy)
        detector_instance = detector_cls(
            mmd_estimator=estimator_copy,
            training_parameter=train_params,
            dataset_train=dataset_train,
            dataset_validation=dataset_val,
        )

        trainer = create_mmd_trainer(
            trainer_config=request.pytorch_trainer_config,
            trainer_backend=getattr(train_params, "trainer_backend", None),
            use_fused_kernel=getattr(train_params, "use_fused_kernel", None),
        )

        # 5. Fit model
        trainer.fit(model=detector_instance)

        # 6. Extract trained ARD weights
        trained_vars = detector_instance.get_trained_variables()
        ard_weights_tensor = trained_vars.ard_weights_kernel_k.detach().cpu()
        ard_weights_clamped = torch.clamp(ard_weights_tensor, min=0.0).flatten().tolist()

        nan_ratio = trained_vars.training_stats.nan_ratio if trained_vars.training_stats else 0.0
        is_success = bool(nan_ratio is None or nan_ratio < 0.9)
        loss_final = 0.0
        if trained_vars.trajectory_record_training and len(trained_vars.trajectory_record_training) > 0:
            loss_final = float(trained_vars.trajectory_record_training[-1].loss)
        # end if
        execution_duration = time.time() - start_wall_time

        return PathAggregationTaskResult(
            task_id=request.task_id,
            lambda_val=request.lambda_val,
            split_index=request.split_index,
            is_success=is_success,
            trained_ard_weights=ard_weights_clamped,
            loss_final=loss_final,
            execution_time_seconds=execution_duration,
        )
    except Exception as error:
        logger.error(f"Task {request.task_id} failed with error: {error}")
        execution_duration = time.time() - start_wall_time
        dim_size = request.estimator.kernel_obj.ard_weights.shape[0]
        return PathAggregationTaskResult(
            task_id=request.task_id,
            lambda_val=request.lambda_val,
            split_index=request.split_index,
            is_success=False,
            trained_ard_weights=[0.0] * dim_size,
            loss_final=0.0,
            execution_time_seconds=execution_duration,
        )
    # end try
# end def


# Alias for compatibility with implementation plan
worker_path_aggregation_optimization_routine = execute_optimization_task_path_aggregation
