from dataclasses import dataclass
from typing import List
import typing as ty

from .module_configs.resource_config import ResourceConfigArgs
from .module_configs.approach_config import ApproachConfigArgs
from .module_configs.dataset_config import DataSetConfigArgs
from .module_configs.algorithm_configs.algorithm_config import (
    AlgorithmOneConfigArgs,
    CvSelectionConfigArgs,
    BaselineMmdConfigArgs,
    LinearVariableSelectionConfigArgs
)
from .module_configs.algorithm_configs.module_optimisation_config import (
    MmdOptimisationConfigTemplate,
    ConfigTPamiDraft,
    ConfigRapid,
)


import typing as ty
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import torch


from ..logger_unit import handler

from ..baselines.regression_based_variable_selection.tst_based_regression_tuner import TstBasedRegressionTunerResult
from ..detection_algorithm.cross_validation_detector.cross_validation_detector import CrossValidationTrainedParameter
from ..detection_algorithm.detection_algorithm_one import AlgorithmOneResult

logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)



@dataclass
class DetectorAlgorithmConfigArgs:
    mmd_cv_selection_args: ty.Optional[ty.Union[str, ty.Dict, CvSelectionConfigArgs]] = None
    mmd_algorithm_one_args: ty.Optional[ty.Union[str, ty.Dict, AlgorithmOneConfigArgs]] = None
    mmd_baseline_args: ty.Optional[ty.Union[str, ty.Dict, BaselineMmdConfigArgs]] = None
    linear_variable_selection_args: ty.Optional[ty.Union[str, ty.Dict, LinearVariableSelectionConfigArgs]] = None
    mmd_optimiser_configs: MmdOptimisationConfigTemplate = ConfigTPamiDraft()

    def __post_init__(self):
        if self.mmd_baseline_args == '':
            self.mmd_baseline_args = None
        elif isinstance(self.mmd_baseline_args, dict):
            self.mmd_baseline_args = BaselineMmdConfigArgs(**self.mmd_baseline_args)
        # end if

        if self.mmd_cv_selection_args == '':
            self.mmd_cv_selection_args = None
        elif isinstance(self.mmd_cv_selection_args, dict):
            self.mmd_cv_selection_args = CvSelectionConfigArgs(**self.mmd_cv_selection_args)
        # end if
        if self.mmd_algorithm_one_args == '':
            self.mmd_algorithm_one_args = None
        elif isinstance(self.mmd_algorithm_one_args, dict):
            self.mmd_algorithm_one_args = AlgorithmOneConfigArgs(**self.mmd_algorithm_one_args)
        # end if
        #  
        if self.linear_variable_selection_args == '':
            self.linear_variable_selection_args = None
        elif isinstance(self.linear_variable_selection_args, dict):
            self.linear_variable_selection_args = LinearVariableSelectionConfigArgs(**self.linear_variable_selection_args)
        # end if


@dataclass
class InterfaceConfigArgs:
    """The class for configuration of `Interface`.
    
    Parameters
    ----------
    resource_config_args: ResourceConfigArgs
        Configuration for resource.
    approach_config_args: ApproachConfigArgs
        Configuration for approach.
    data_config_args: DataSetConfigArgs
        Configuration for dataset.
    detector_algorithm_config_args: DetectorAlgorithmConfigArgs
        Configuration for detector algorithm.
    """
    resource_config_args: ResourceConfigArgs
    approach_config_args: ApproachConfigArgs
    data_config_args: DataSetConfigArgs
    detector_algorithm_config_args: DetectorAlgorithmConfigArgs
    
    def __post_init__(self):
        # reset the distributed backend.
        if isinstance(self.detector_algorithm_config_args.mmd_algorithm_one_args, AlgorithmOneConfigArgs):
            self.detector_algorithm_config_args.mmd_algorithm_one_args.parameter_search_parameter.backend = self.resource_config_args.dask_config_detection.distributed_mode
        if isinstance(self.detector_algorithm_config_args.mmd_cv_selection_args, CvSelectionConfigArgs):
            self.detector_algorithm_config_args.mmd_cv_selection_args.parameter_search_parameter.backend = self.resource_config_args.dask_config_detection.distributed_mode
        # end if
