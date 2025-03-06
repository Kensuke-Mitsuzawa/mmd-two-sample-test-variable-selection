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
from ..detection_algorithm.baseline_mmd import BaselineMmdResult

from .interface_config_args import InterfaceConfigArgs

logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


# --------------------------------------------------------------------------- #
# Return Object definition.

import json

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return obj.as_posix()
        if isinstance(obj, np.ndarray):
            return obj.tolist()        

@dataclass
class BasicVariableSelectionResult:
    """Basic result of variable selection.
    """
    weights: ty.Union[ty.List[int], np.ndarray]
    variables: ty.List[int]
    p_value: float
    verbose_field: ty.Optional[ty.Union[TstBasedRegressionTunerResult, BaselineMmdResult, CrossValidationTrainedParameter, AlgorithmOneResult]] = None
    n_sample_training: ty.Optional[int] = None
    n_sample_test: ty.Optional[int] = None
    
    def __post_init__(self):
        if isinstance(self.weights, list):
            self.weights = np.array(self.weights)
        # end if
        
    def as_dict(self):
        """Converting into a dictionary object, expect `verbose_field` field.
        """
        if isinstance(self.weights, torch.Tensor):
            self.weights = self.weights.detach().cpu().numpy()
        # end if
        assert isinstance(self.weights, np.ndarray)
        d_obj = dict(
            weights=self.weights.tolist(),
            variables=self.variables,
            p_value=self.p_value,
            n_sample_training=self.n_sample_training,
            n_sample_test=self.n_sample_test
        )
        return d_obj


@dataclass
class OutputObject:
    configurations: InterfaceConfigArgs
    detection_result_sample_based: ty.Optional[BasicVariableSelectionResult]
    # TODO
    # detection_result_time_slicing: ty.Optional[ty.List[BasicVariableSelectionResult]]
    
    def as_json(self) -> str:
        if isinstance(self.detection_result_sample_based, BasicVariableSelectionResult):
            detection_result = asdict(self.detection_result_sample_based)
        # elif isinstance(self.detection_result_time_slicing, list):
        #     detection_result =[asdict(__d) for __d in self.detection_result_time_slicing]
        else:
            raise ValueError(f'Unknown type case.')
        # end if
        
        dict_obj = {
            'configurations': asdict(self.configurations),
            'detection_result': detection_result
        }
        return json.dumps(dict_obj, cls=JSONEncoder)


