import time
import timeit
import logging
import typing
import typing as ty
import traceback

import torch
import pytorch_lightning as pl
import numpy as np
import ot

from ...datasets.base import BaseDataset
from ...datasets.file_onetime_load_backend_static_dataset import FileBackendOneTimeLoadStaticDataset
from ...mmd_estimator.mmd_estimator import BaseMmdEstimator
from ...utils.permutation_test_runner import PermutationTest
from ...utils.variable_detection import detect_variables
from ...logger_unit import handler
from ..interpretable_mmd_detector import InterpretableMmdDetector
from ..utils.permutation_tests import permutation_tests
from ..commons import (
    InterpretableMmdTrainParameters, 
    RegularizationParameter,
)
from .commons import (
    SubLearnerTrainingResult,
    RequestDistributedFunction,
    CrossValidationAlgorithmParameter,
    AggregationKey
)

logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


# -------------------------------------------------------------
# util functions


def scale_ard_weights(ard_weights_power2: torch.Tensor) -> torch.Tensor:
    """Compute the scaled ARD weights that is ranged in [0, 1.0]
    """
    return ard_weights_power2 / torch.max(ard_weights_power2)


def select_threshold(
        ard_weight_square: torch.Tensor,
        threshold_mode: str,
        threshold_value: float) -> typing.Tuple[torch.Tensor, float]:
    """Selecting the predicted coordinates for the given threshold.
    The recommended choice is threshold_mode = 'normalized_min' and threshold_value = 0.01,
    or threshold_mode = 'incremental' and threshold_value = 0.001.
    Args:
        threshold_mode: possible choices: min, normalized_min, mean, incremental
        threshold_value: threshold value or incremental delta value
    Return: (ARD weights, threshold value)
    """
    assert threshold_mode in ('min', 'normalized_min', 'mean')

    if threshold_mode == 'normalized_min':
        weight_power_2 = scale_ard_weights(ard_weight_square)
    else:
        weight_power_2 = ard_weight_square
    # end if

    if threshold_mode == 'mean':
        threshold_ = torch.mean(weight_power_2)
    elif threshold_mode == 'median':
        threshold_ = torch.median(weight_power_2)
    elif (threshold_mode == 'min') or (threshold_mode == 'normalized_min'):
        assert threshold_value > 0, f'{threshold_value} must be greater than 0.0'
        threshold_ = threshold_value
    else:
        raise NotImplementedError(f'{threshold_mode} does not exist.')
    # end if

    return weight_power_2, float(threshold_)


def get_frequency_tensor(
        ard_weight_square: torch.Tensor,
        threshold_mode: str = 'normalized_min',
        threshold_value: float = 0.1) -> torch.Tensor:
    # filtering effective ARD weights
    ard_weights, threshold_ = select_threshold(ard_weight_square, threshold_mode, threshold_value)
    ard_weight_selected_binary = torch.where(ard_weights > threshold_, 1.0, 0.0)

    return torch.tensor(ard_weight_selected_binary)


# ---------------------------------------------------------------------------------------------


def dask_worker_script(args: RequestDistributedFunction) -> SubLearnerTrainingResult:
    """A function that Dask workers calls.
    :param args: 6 elements. See the asserting message below.
    :return: (task_id, trained-result)
    """
    from mmd_tst_variable_detector.exceptions import OptimizationException
    task_id: AggregationKey = args.task_id
    training_parameter: InterpretableMmdTrainParameters = args.training_parameter
    __dataset_train: BaseDataset = args.dataset_train
    __dataset_val: BaseDataset = args.dataset_val
    trainer_lightning: pl.Trainer = args.trainer_lightning
    mmd_estimator: BaseMmdEstimator = args.mmd_estimator
    ss_algorithm_param: CrossValidationAlgorithmParameter = args.stability_algorithm_param
    
    # start counting execution time
    start_cpu_time = time.process_time()
    start_wall_time = timeit.default_timer()
    
    init_ard_weights = torch.ones(mmd_estimator.kernel_obj.ard_weights.shape)
    mmd_estimator.kernel_obj.ard_weights = torch.nn.Parameter(init_ard_weights)
    
    if __dataset_train.is_dataset_on_ram():
        dataset_train = __dataset_train.generate_dataset_on_ram()
    else:
        dataset_train = __dataset_train.copy_dataset()
    # end if
        
    if __dataset_val.is_dataset_on_ram():
        dataset_val = __dataset_val.generate_dataset_on_ram()
    else:
        dataset_val = __dataset_val.copy_dataset()    
    # end if
    
    try:
        variable_detector = InterpretableMmdDetector(mmd_estimator=mmd_estimator,
                                               training_parameter=training_parameter,
                                               dataset_train=dataset_train,
                                               dataset_validation=dataset_val)
        # variable_detector = torch.compile(variable_detector)
        trainer_lightning.fit(variable_detector)
    except OptimizationException as e:
        msg_traceback = traceback.format_exc()
        logger.error(f'OptimizationException with {msg_traceback}')
        return SubLearnerTrainingResult(
            job_id=task_id,
            training_parameter=training_parameter,
            training_result=None,
            p_value_selected=None,
            variable_detected=None)
    # end try
    weights_detector_result = variable_detector.get_trained_variables()

    variables = detect_variables(
        variable_detection_approach=ss_algorithm_param.ard_weight_selection_strategy,
        variable_weights=weights_detector_result.ard_weights_kernel_k,
        threshold_weights=ss_algorithm_param.ard_weight_minimum,
        is_normalize_ard_weights=True)

    seq_permutation_result_dev = permutation_tests(
        dataset_val,
        variable_selection_approach='hard',
        interpretable_mmd_result=weights_detector_result,
        distance_functions=(ss_algorithm_param.permutation_test_metric,),
        dask_client=None,
        n_permutation_test=ss_algorithm_param.n_permutation_test)
    p_value_max_dev = max([__res_p.p_value for __res_p in seq_permutation_result_dev])
    
    end_cpu_time = time.process_time()
    end_wall_time = timeit.default_timer()
    
    exec_time_cpu = end_cpu_time - start_cpu_time
    exec_time_wallclock = end_wall_time - start_wall_time
    
    epoch_size = weights_detector_result.trajectory_record_training[-1].epoch
        
    return SubLearnerTrainingResult(
        job_id=task_id,
        training_parameter=training_parameter,
        training_result=weights_detector_result,
        p_value_selected=p_value_max_dev,
        variable_detected=variables,
        execution_time_cpu=exec_time_cpu,
        execution_time_wallclock=exec_time_wallclock,
        epoch=epoch_size)