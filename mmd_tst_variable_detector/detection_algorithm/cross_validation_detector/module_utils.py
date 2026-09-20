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


from ...accelerator_optimizations.worker import worker_execution_routine


def dask_worker_script(args: RequestDistributedFunction) -> SubLearnerTrainingResult:
    """A function that Dask workers call, delegating to worker_execution_routine."""
    return worker_execution_routine(args)