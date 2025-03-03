from dataclasses import dataclass
from typing import List
import typing as ty

from .module_configs.resource_config import ResourceConfigArgs
from .module_configs.approach_config import ApproachConfigArgs
from .module_configs.dataset_config import DataSetConfigArgs
from .module_configs.algorithm_configs.algorithm_config import (
    AlgorithmOneConfigArgs,
    CvSelectionConfigArgs,
    LinearVariableSelectionConfigArgs
)
from .module_configs.algorithm_configs.module_mmd_config import (
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
    mmd_cv_selection_args: ty.Optional[ty.Union[str, CvSelectionConfigArgs]] = None
    mmd_algorithm_one_args: ty.Optional[ty.Union[str, AlgorithmOneConfigArgs]] = None
    linear_variable_selection_args: ty.Optional[ty.Union[str, LinearVariableSelectionConfigArgs]] = None
    mmd_optimiser_configs: MmdOptimisationConfigTemplate = ConfigTPamiDraft()

    def __post_init__(self):
        if self.mmd_cv_selection_args == '':
            self.mmd_cv_selection_args = None
        if self.mmd_algorithm_one_args == '':
            self.mmd_algorithm_one_args = None 
        if self.linear_variable_selection_args == '':
            self.linear_variable_selection_args = None
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



# --------------------------------------------------------------------------- #
# Return Object definition.

# import json

# class JSONEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, Path):
#             return obj.as_posix()
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()        

# @dataclass
# class BasicVariableSelectionResult:
#     """Basic result of variable selection.
#     """
#     weights: ty.Union[ty.List[int], np.ndarray]
#     variables: ty.List[int]
#     p_value: float
#     verbose_field: ty.Optional[ty.Union[TstBasedRegressionTunerResult, CrossValidationTrainedParameter, AlgorithmOneResult]] = None
#     n_sample_training: ty.Optional[int] = None
#     n_sample_test: ty.Optional[int] = None
    
#     def __post_init__(self):
#         if isinstance(self.weights, list):
#             self.weights = np.array(self.weights)
#         # end if
        
#     def as_dict(self):
#         """Converting into a dictionary object, expect `verbose_field` field.
#         """
#         if isinstance(self.weights, torch.Tensor):
#             self.weights = self.weights.detach().cpu().numpy()
#         # end if
#         assert isinstance(self.weights, np.ndarray)
#         d_obj = dict(
#             weights=self.weights.tolist(),
#             variables=self.variables,
#             p_value=self.p_value,
#             n_sample_training=self.n_sample_training,
#             n_sample_test=self.n_sample_test
#         )
#         return d_obj


# @dataclass
# class OutputObject:
#     configurations: InterfaceConfigArgs
#     detection_result_sample_based: ty.Optional[BasicVariableSelectionResult]
#     # TODO
#     # detection_result_time_slicing: ty.Optional[ty.List[BasicVariableSelectionResult]]
    
#     def as_json(self) -> str:
#         if isinstance(self.detection_result_sample_based, BasicVariableSelectionResult):
#             detection_result = asdict(self.detection_result_sample_based)
#         # elif isinstance(self.detection_result_time_slicing, list):
#         #     detection_result =[asdict(__d) for __d in self.detection_result_time_slicing]
#         else:
#             raise ValueError(f'Unknown type case.')
#         # end if
        
#         dict_obj = {
#             'configurations': asdict(self.configurations),
#             'detection_result': detection_result
#         }
#         return json.dumps(dict_obj, cls=JSONEncoder)


