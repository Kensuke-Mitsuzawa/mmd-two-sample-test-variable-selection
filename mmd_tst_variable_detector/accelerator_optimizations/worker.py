import time
import timeit
import logging
import traceback
import typing as ty

import torch
import pytorch_lightning as pl

from ..datasets.base import BaseDataset
from ..mmd_estimator.mmd_estimator import BaseMmdEstimator
from ..utils.variable_detection import detect_variables
from ..logger_unit import handler
from ..detection_algorithm.interpretable_mmd_detector import InterpretableMmdDetector
from ..detection_algorithm.utils.permutation_tests import permutation_tests
from ..detection_algorithm.commons import (
    InterpretableMmdTrainParameters,
)
from ..detection_algorithm.cross_validation_detector.commons import (
    SubLearnerTrainingResult,
    RequestDistributedFunction,
    CrossValidationAlgorithmParameter,
    AggregationKey,
)
from ..detection_algorithm.pytorch_lightning_trainer import get_mmd_detector_class
from ..exceptions import OptimizationException

logger = logging.getLogger(f"{__package__}.{__name__}")
logger.addHandler(handler)


def worker_execution_routine(args: RequestDistributedFunction) -> SubLearnerTrainingResult:
    """Standardized worker execution routine for MMD optimization tasks.

    Parameters
    ----------
    args : RequestDistributedFunction
        Task arguments containing datasets, estimators, trainer and parameters.

    Returns
    -------
    SubLearnerTrainingResult
        Optimization result including trained weights, detected variables, and metrics.
    """
    task_id: AggregationKey = args.task_id
    training_parameter: InterpretableMmdTrainParameters = args.training_parameter
    __dataset_train: BaseDataset = args.dataset_train
    __dataset_val: BaseDataset = args.dataset_val
    trainer_lightning: pl.Trainer = args.trainer_lightning
    mmd_estimator: BaseMmdEstimator = args.mmd_estimator
    ss_algorithm_param: CrossValidationAlgorithmParameter = args.stability_algorithm_param

    start_cpu_time = time.process_time()
    start_wall_time = timeit.default_timer()

    init_ard_weights = torch.ones(mmd_estimator.kernel_obj.ard_weights.shape)
    mmd_estimator.kernel_obj.ard_weights = torch.nn.Parameter(init_ard_weights)

    if __dataset_train.is_dataset_on_ram():
        dataset_train = __dataset_train.generate_dataset_on_ram()
    else:
        dataset_train = __dataset_train.copy_dataset()

    if __dataset_val.is_dataset_on_ram():
        dataset_val = __dataset_val.generate_dataset_on_ram()
    else:
        dataset_val = __dataset_val.copy_dataset()

    try:
        use_legacy = getattr(training_parameter, "use_legacy_optimization", False) or \
                     getattr(trainer_lightning, "use_legacy_optimization", False)
        detector_cls = get_mmd_detector_class(use_legacy_optimization=use_legacy)
        variable_detector = detector_cls(
            mmd_estimator=mmd_estimator,
            training_parameter=training_parameter,
            dataset_train=dataset_train,
            dataset_validation=dataset_val,
        )
        trainer_lightning.fit(variable_detector)
    except OptimizationException:
        msg_traceback = traceback.format_exc()
        logger.error(f"OptimizationException with {msg_traceback}")
        return SubLearnerTrainingResult(
            job_id=task_id,
            training_parameter=training_parameter,
            training_result=None,
            p_value_selected=None,
            variable_detected=None,
        )

    weights_detector_result = variable_detector.get_trained_variables()

    variables = detect_variables(
        variable_detection_approach=ss_algorithm_param.ard_weight_selection_strategy,
        variable_weights=weights_detector_result.ard_weights_kernel_k,
        threshold_weights=ss_algorithm_param.ard_weight_minimum,
        is_normalize_ard_weights=True,
    )

    seq_permutation_result_dev = permutation_tests(
        dataset_val,
        variable_selection_approach="hard",
        interpretable_mmd_result=weights_detector_result,
        distance_functions=(ss_algorithm_param.permutation_test_metric,),
        dask_client=None,
        n_permutation_test=ss_algorithm_param.n_permutation_test,
    )
    p_value_max_dev = max([__res_p.p_value for __res_p in seq_permutation_result_dev])

    end_cpu_time = time.process_time()
    end_wall_time = timeit.default_timer()

    exec_time_cpu = end_cpu_time - start_cpu_time
    exec_time_wallclock = end_wall_time - start_wall_time

    epoch_size = (
        weights_detector_result.trajectory_record_training[-1].epoch
        if weights_detector_result.trajectory_record_training
        else 0
    )

    return SubLearnerTrainingResult(
        job_id=task_id,
        training_parameter=training_parameter,
        training_result=weights_detector_result,
        p_value_selected=p_value_max_dev,
        variable_detected=variables,
        execution_time_cpu=exec_time_cpu,
        execution_time_wallclock=exec_time_wallclock,
        epoch=epoch_size,
    )
