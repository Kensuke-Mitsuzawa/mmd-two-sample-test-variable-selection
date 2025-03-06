import typing as ty
import shutil
import time
import logging
from copy import deepcopy
from pathlib import Path
from distributed import Client
from tempfile import mkdtemp
from dataclasses import dataclass

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger

from .. import logger_unit
# detector class sample selection based detector
from ..mmd_estimator import BaseMmdEstimator
from ..datasets import BaseDataset
from ..utils import (
    PostProcessLoggerHandler, 
    detect_variables, 
    PermutationTest)

from .pytorch_lightning_trainer import PytorchLightningDefaultArguments
from .interpretable_mmd_detector import (
    InterpretableMmdTrainResult, 
    InterpretableMmdTrainParameters)
from .interpretable_mmd_detector import InterpretableMmdDetector


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(logger_unit.handler)


@dataclass
class BaselineMmdResult:
    selected_variables: ty.List[int]
    trained_ard_weights: ty.Optional[ty.Union[np.ndarray, torch.Tensor]]
    p_value_test: ty.Optional[float] = None
    pl_loggers: ty.Optional[ty.List[Logger]] = None  # saving logger object. So, a user can do logging operation later.
    interpretable_mmd_train_result: ty.Optional[InterpretableMmdTrainResult] = None
    


def baseline_mmd(
    mmd_estimator: BaseMmdEstimator,
    pytorch_trainer_config: PytorchLightningDefaultArguments,
    base_training_parameter: InterpretableMmdTrainParameters,
    dataset_training: BaseDataset,
    dataset_dev: ty.Optional[BaseDataset] = None,
    dataset_test: ty.Optional[BaseDataset] = None,
    path_work_dir: ty.Optional[Path] = None,
    permutation_test_runner: ty.Optional[PermutationTest] = None,
    post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
    # test_distance_functions: ty.Tuple[str, ...] = ('sliced_wasserstein',),
    n_permutation_test: int = 500
    ) -> BaselineMmdResult:
    """
    """
    variable_detector = InterpretableMmdDetector(
        mmd_estimator=deepcopy(mmd_estimator),
        training_parameter=base_training_parameter,
        dataset_train=dataset_training,
        dataset_validation=dataset_training if dataset_dev is None else dataset_dev,)
    pl_trainer_obj = pl.Trainer(**pytorch_trainer_config.as_dict())
    pl_trainer_obj.fit(variable_detector)
    
    detection_result_obj = variable_detector.get_trained_variables()
    variable_detected = detect_variables(detection_result_obj.ard_weights_kernel_k)

    # conducting permutation test
    if dataset_test is not None:
        dataset_test_selected = dataset_test.get_selected_variables_dataset(tuple(variable_detected))
        if permutation_test_runner is None:
            permutation_test_obj = PermutationTest(n_permutation_test=n_permutation_test,)
        else:
            permutation_test_obj = permutation_test_runner
        # end
        p_value, stats_permutation_test = permutation_test_obj.run_test(dataset=dataset_test_selected)
    else:
        p_value, stats_permutation_test = None, None
    # end if
    
    if post_process_handler is not None:
        __run_name = f'baseline_mmd_{time.time()}'
        __loggers = post_process_handler.initialize_logger(run_name=__run_name, group_name='baseline_mmd')
        post_process_handler.log(loggers=__loggers, target_object=detection_result_obj)
    # end if

    return BaselineMmdResult(
        selected_variables=variable_detected,
        trained_ard_weights=detection_result_obj.ard_weights_kernel_k,
        interpretable_mmd_train_result=detection_result_obj,
        p_value_test=p_value
    )
