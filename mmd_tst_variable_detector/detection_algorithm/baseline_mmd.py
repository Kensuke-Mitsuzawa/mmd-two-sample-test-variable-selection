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

from .base import BaseVariableDetector
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


class BaselineMmdVariableDetector(BaseVariableDetector):
    """Baseline MMD optimization variable detector."""

    def __init__(
        self,
        estimator: BaseMmdEstimator,
        training_parameter: InterpretableMmdTrainParameters,
        pytorch_trainer_config: ty.Optional[PytorchLightningDefaultArguments] = None,
        post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
        dask_client: ty.Optional[Client] = None,
        dataset_test: ty.Optional[BaseDataset] = None,
        permutation_test_runner: ty.Optional[PermutationTest] = None,
        n_permutation_test: int = 500,
        path_work_dir: ty.Optional[Path] = None,
        **kwargs: ty.Any,
    ) -> None:
        super().__init__(
            estimator=estimator,
            pytorch_trainer_config=pytorch_trainer_config,
            post_process_handler=post_process_handler,
            dask_client=dask_client,
            **kwargs,
        )
        self.training_parameter = training_parameter
        self.dataset_test = dataset_test
        self.permutation_test_runner = permutation_test_runner
        self.n_permutation_test = n_permutation_test
        self.path_work_dir = path_work_dir
    # end def

    def run_detection(
        self,
        training_dataset: BaseDataset,
        validation_dataset: ty.Optional[BaseDataset] = None,
        dataset_test: ty.Optional[BaseDataset] = None,
        **kwargs: ty.Any,
    ) -> BaselineMmdResult:
        """Run baseline MMD optimization for variable detection."""
        test_data = dataset_test if dataset_test is not None else self.dataset_test

        variable_detector = InterpretableMmdDetector(
            mmd_estimator=deepcopy(self.estimator),
            training_parameter=self.training_parameter,
            dataset_train=training_dataset,
            dataset_validation=training_dataset if validation_dataset is None else validation_dataset,
        )
        trainer_config = self.pytorch_trainer_config if self.pytorch_trainer_config is not None else PytorchLightningDefaultArguments()
        pl_trainer_obj = pl.Trainer(**trainer_config.as_dict())
        pl_trainer_obj.fit(variable_detector)

        detection_result_obj = variable_detector.get_trained_variables()
        variable_detected = detect_variables(detection_result_obj.ard_weights_kernel_k)

        # conducting permutation test
        if test_data is not None:
            dataset_test_selected = test_data.get_selected_variables_dataset(tuple(variable_detected))
            if self.permutation_test_runner is None:
                permutation_test_obj = PermutationTest(n_permutation_test=self.n_permutation_test)
            else:
                permutation_test_obj = self.permutation_test_runner
            # end if
            p_value, stats_permutation_test = permutation_test_obj.run_test(dataset=dataset_test_selected)
        else:
            p_value, stats_permutation_test = None, None
        # end if

        if self.post_process_handler is not None:
            __run_name = f'baseline_mmd_{time.time()}'
            __loggers = self.post_process_handler.initialize_logger(run_name=__run_name, group_name='baseline_mmd')
            self.post_process_handler.log(loggers=__loggers, target_object=detection_result_obj)
        # end if

        return BaselineMmdResult(
            selected_variables=variable_detected,
            trained_ard_weights=detection_result_obj.ard_weights_kernel_k,
            interpretable_mmd_train_result=detection_result_obj,
            p_value_test=p_value,
        )
    # end def
# end class


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
    """Execute baseline MMD variable detection via BaselineMmdVariableDetector."""
    detector = BaselineMmdVariableDetector(
        estimator=mmd_estimator,
        training_parameter=base_training_parameter,
        pytorch_trainer_config=pytorch_trainer_config,
        post_process_handler=post_process_handler,
        dataset_test=dataset_test,
        permutation_test_runner=permutation_test_runner,
        n_permutation_test=n_permutation_test,
        path_work_dir=path_work_dir,
    )
    return detector.run_detection(
        training_dataset=dataset_training,
        validation_dataset=dataset_dev,
        dataset_test=dataset_test,
    )
# end def

