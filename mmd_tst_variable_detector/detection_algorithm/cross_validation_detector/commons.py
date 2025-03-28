import logging
import typing
import typing as ty

import torch
import numpy as np
import pytorch_lightning as pl

from dataclasses import dataclass

from ...datasets import BaseDataset
from ...logger_unit import handler

from ..search_regularization_min_max import RegularizationSearchParameters

from ..commons import (
    RegularizationParameter, 
    InterpretableMmdTrainParameters, 
    InterpretableMmdTrainResult)


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)

# ------------------------------------------------------------
# parameter objects for Cross-Validation approach.


SAMPLING_STRATEGY = ('cross-validation', 'random-splitting', 'k-fold-cross-validation')
STABILITY_SCORE_BASE = ('ard', 'variable')
WEIGHTING_MODE = ('plane', 'p_value', 'test_power', 'p_value_filter', 'p_value_filter_test_power', 'p_value_min_test_power')
PRE_FILTERING_ESTIMATORS = ('off', 'ranking_top_k', 'ranking_top_ratio')
APPROACH_REGULARIZATION_PARAMETER = ('fixed_range', 'param_searching')


# Key Object that Cross-Validation executor and Sub-learner-Executor use in common.

@dataclass
class AggregationKey:    
    approach_regularization_parameter: str
    trial_id_cross_validation: int
    regularization: ty.Optional[RegularizationParameter] = None
    job_id: ty.Optional[int] = None 

    def __str__(self) -> str:
        return self.get_key_str()
    
    def get_key_str(self) -> str:
        if self.approach_regularization_parameter == 'fixed_range':
            return f"{str(self.regularization)}-{self.job_id}"
        elif self.approach_regularization_parameter == 'param_searching':
            return f"{self.trial_id_cross_validation}-{self.job_id}"
        else:
            raise ValueError(f"Invalid approach_regularization_parameter: {self.approach_regularization_parameter}")
        
    def __post_init__(self):
        assert self.approach_regularization_parameter in APPROACH_REGULARIZATION_PARAMETER

# ------------------------------------------------------------

@dataclass
class DistributedComputingParameter:
    """Configurations for distributed computing during Cross-Validation approach.
    """
    job_batch_size: int = 5


@dataclass
class CrossValidationAlgorithmParameter(object):
    approach_regularization_parameter: str = "fixed_range"
    candidate_regularization_parameter: ty.Union[str, ty.List[RegularizationParameter]] = "auto"
    regularization_search_parameter: RegularizationSearchParameters = RegularizationSearchParameters()
    n_subsampling: int = 5  # number of subsampling, this is K-fold when sampling_strategy is "fold-cross-validation"
    sampling_strategy: str = 'cross-validation'  # "subsampling", "bootstrap", "cross-validation", "fold-cross-validation"
    ratio_subsampling: float = 0.8  # ratio to training data for cross-validation mode.
    strategy_stability_score: str = 'mean'
    threshold_stability_score: float = 0.1
    weighting_mode: str = 'p_value_min_test_power'
    stability_score_base: str = 'ard'  # ard or variable
    ard_weight_minimum: float = 0.1  # will be deprecated
    ard_weight_selection_strategy: str = 'hist_based'
    is_normalize_agg_stability_score: bool = True
    is_weight_stability_score: bool = True
    permutation_test_metric: str = 'sliced_wasserstein'
    n_permutation_test: int = 500
    is_attempt_all_weighting: bool = True
    # -------------------------------------------
    # for pre-filtering
    # This option enables to pick up trained MMD estimators before the aggrgation operation.
    # By doing that, we can cut off MMD estimators of which detection results were not good.
    # See: https://github.com/Kensuke-Mitsuzawa/mmd-tst-variable-detector/issues/394
    pre_filtering_trained_estimator: str = 'off'
    pre_filtering_parameter: ty.Union[int, float] = 0.3

    def __post_init__(self):
        assert self.approach_regularization_parameter in APPROACH_REGULARIZATION_PARAMETER
        assert self.sampling_strategy in SAMPLING_STRATEGY, f"Invalid sampling_strategy: {self.sampling_strategy}"
        assert self.stability_score_base in STABILITY_SCORE_BASE, f"Invalid stability_score_base: {self.stability_score_base}"
        assert self.weighting_mode in WEIGHTING_MODE, f"Invalid weighting_mode: {self.weighting_mode}"
        assert self.permutation_test_metric == 'sliced_wasserstein'
        assert self.pre_filtering_trained_estimator in PRE_FILTERING_ESTIMATORS, f"Invalid pre_filtering_trained_estimator: {self.pre_filtering_trained_estimator}"
        
        if self.pre_filtering_trained_estimator == 'ranking_top_k':
            assert isinstance(self.pre_filtering_parameter, int), "pre_filtering_parameter must be an integer."
        elif self.pre_filtering_trained_estimator == 'ranking_top_ratio':
            assert isinstance(self.pre_filtering_parameter, float), "pre_filtering_parameter must be a float."
        else:
            pass
        # end if

        if isinstance(self.candidate_regularization_parameter, list):
            assert len(self.candidate_regularization_parameter) > 0, "No parameter is given."
            for __i, o in enumerate(self.candidate_regularization_parameter):
                if isinstance(o, tuple):
                    self.candidate_regularization_parameter[__i] = RegularizationParameter(*o)
        elif isinstance(self.candidate_regularization_parameter, str):
            assert self.candidate_regularization_parameter == 'auto', "candidate_regularization_parameter must be either a list or auto"


@dataclass
class CrossValidationTrainParameters(object):
    """
    Parameters
    ------------
    computation_backend: str
        Either of following choices,
        1. 'single': Single machine
        2. 'dask': Dask distributed computing
        3. 'joblib': Joblib parallel computing
    """
    algorithm_parameter: CrossValidationAlgorithmParameter
    base_training_parameter: InterpretableMmdTrainParameters
    distributed_parameter: DistributedComputingParameter
    # computation_backend = None  # deprecated
    # dist_parameter = None  # deprecated


# -------------------------------------------------------------
# Trained parameter


class ExecutionTimeStatistics(typing.NamedTuple):
    # execution time statistics
    total_execution_time_wallclock: float
    avg_execution_time_wallclock: float
    std_execution_time_wallclock: float
    min_execution_time_wallclock: float
    max_execution_time_wallclock: float
    # epochs
    avg_epochs: float
    std_epochs: float
    min_epochs: float
    max_epochs: float


@dataclass
class AggregationResultContainer:
    """A data container for saving aggregated information.
    Added by a request: https://github.com/Kensuke-Mitsuzawa/mmd-tst-variable-detector/issues/367"""
    weighting_name: str
    stability_score_base: str
    stability_score_matrix: ty.Optional[np.ndarray]  # 2d-array (|reg|, dim)
    array_s_hat: ty.Optional[np.ndarray]  # 1d-array (dim,)
    stable_s_hat: ty.List[int]
    filtering_estimators: ty.Optional[str] = None
    filtering_parameter: ty.Optional[float] = None

    def to_dict(self):
        return dict(
            weighting_name=self.weighting_name,
            stability_score_base=self.stability_score_base,
            stability_score_matrix=self.stability_score_matrix,
            array_s_hat=self.array_s_hat,
            stable_s_hat=self.stable_s_hat,
            filtering_estimators=self.filtering_estimators,
            filtering_parameter=self.filtering_parameter
        )


@dataclass
class SubEstimatorResultContainer:
    """
    Compatible with the result of `SubLearnerTrainingResult`.
    I re-define it since I want to return it in the output.
    """
    job_id: AggregationKey
    training_parameter: InterpretableMmdTrainParameters
    training_result: ty.Optional[InterpretableMmdTrainResult]
    p_value_selected: ty.Optional[float]
    variable_detected: ty.Optional[ty.List[int]]
    ard_weight_selected_binary: ty.Optional[torch.Tensor] = None
    execution_time_wallclock: ty.Optional[float] = None
    execution_time_cpu: ty.Optional[float] = None
    epoch: ty.Optional[int] = None
    
    def get_job_id_string(self) -> str:
        return self.job_id.get_key_str()
    
    def to_dict(self):
        return dict(
            job_id=self.job_id.get_key_str(),
            training_parameter=self.training_parameter.__dict__,
            training_result=self.training_result.to_dict() if self.training_result is not None else None,
            p_value_selected=self.p_value_selected,
            variable_detected=self.variable_detected,
            ard_weight_selected_binary=self.ard_weight_selected_binary,
            execution_time_wallclock=self.execution_time_wallclock,
            execution_time_cpu=self.execution_time_cpu,
            epoch=self.epoch
        )


@dataclass
class CrossValidationTrainedParameter:
    regularization: ty.List[RegularizationParameter]
    stability_score_matrix: ty.Optional[np.ndarray]  # 2d-array (|reg|, dim)
    array_s_hat: ty.Optional[np.ndarray]  # 1d-array (dim,)
    stable_s_hat: ty.List[int]
    variable_detection_postprocess_hard: ty.Optional[InterpretableMmdTrainResult] = None
    variable_detection_postprocess_soft: ty.Optional[InterpretableMmdTrainResult] = None
    execution_time_statistics: ty.Optional[ExecutionTimeStatistics] = None
    training_parameters: ty.Optional[CrossValidationTrainParameters] = None
    seq_aggregation_results: ty.Optional[ty.List[AggregationResultContainer]] = None # I use this field only when weighting_mode is `all`.
    seq_sub_estimators: ty.Optional[ty.List[SubEstimatorResultContainer]] = None
    
    def to_dict(self):
        """Making this object serializable.
        """
        exec_time_stats = self.execution_time_statistics._asdict() if self.execution_time_statistics is not None else None
        
        dict_obj = dict(
            regularization=[__t._asdict() for __t in self.regularization],
            stability_score_matrix=self.stability_score_matrix,
            array_s_hat=self.array_s_hat,
            stable_s_hat=self.stable_s_hat,
            variable_detection_postprocess_hard=self.variable_detection_postprocess_hard.to_dict() if self.variable_detection_postprocess_hard is not None else None,
            variable_detection_postprocess_soft=self.variable_detection_postprocess_soft.to_dict() if self.variable_detection_postprocess_soft is not None else None,
            execution_time_statistics=exec_time_stats,
            training_parameters=self.training_parameters.__dict__ if self.training_parameters is not None else None,
            seq_aggregation_results=[__t.to_dict() for __t in self.seq_aggregation_results] if self.seq_aggregation_results is not None else None,
            seq_sub_estimators=[__obj.to_dict() for __obj in self.seq_sub_estimators] if self.seq_sub_estimators is not None else None
        )
        
        return dict_obj


# ------------------------------------------------------------
# parameter objects that I use internally.


@dataclass
class RequestDistributedFunction:
    __slots__ = ('task_id', 'training_parameter', 
                 'dataset_train', 'dataset_val', 
                 'trainer_lightning', 'mmd_estimator', 'stability_algorithm_param')
    
    task_id: AggregationKey
    training_parameter: InterpretableMmdTrainParameters
    dataset_train: BaseDataset
    dataset_val: BaseDataset
    trainer_lightning: pl.Trainer
    mmd_estimator: "BaseMmdEstimator"  # type: ignore
    stability_algorithm_param: CrossValidationAlgorithmParameter


@dataclass
class SubLearnerTrainingResult:
    job_id: AggregationKey
    training_parameter: InterpretableMmdTrainParameters
    training_result: ty.Optional[InterpretableMmdTrainResult]
    p_value_selected: ty.Optional[float]
    variable_detected: ty.Optional[ty.List[int]]
    ard_weight_selected_binary: ty.Optional[torch.Tensor] = None
    execution_time_wallclock: ty.Optional[float] = None
    execution_time_cpu: ty.Optional[float] = None
    epoch: ty.Optional[int] = None
    
    def get_job_id_string(self) -> str:
        return self.job_id.get_key_str()
    
    def convert2SubEstimatorResultContainer(self) -> SubEstimatorResultContainer:
        return SubEstimatorResultContainer(
            job_id=self.job_id,
            training_parameter=self.training_parameter,
            training_result=self.training_result,
            p_value_selected=self.p_value_selected,
            variable_detected=self.variable_detected,
            ard_weight_selected_binary=self.ard_weight_selected_binary,
            execution_time_wallclock=self.execution_time_wallclock,
            execution_time_cpu=self.execution_time_cpu,
            epoch=self.epoch)
    

class CrossValidationAggregatedResult(typing.NamedTuple):
    stable_s_hat: ty.Optional[ty.List[int]]  # a list of selected coordinates
    array_s_hat: ty.Optional[torch.Tensor]  # a tensor (dim,) computed by avg(stability-score).
    stability_score_matrix: ty.Optional[torch.Tensor]  # a tensor (|reg|, dim)
    learner_training_log: ty.Optional[ty.List[SubLearnerTrainingResult]] # a list of[]
    lambda_labels: ty.Optional[ty.List[str]]  # a list of lambda labels
