import typing as ty
from sympy import im
from dataclasses import dataclass
from pathlib import Path
import torch
import optuna

from ...commons import (
    RegularizationParameter,
    InterpretableMmdTrainResult)


@dataclass
class _DaskFunctionReturn:
    trial: optuna.Trial
    optuna_objective_value: float
    ratio_variable: float
    selected_variables: ty.List[int]
    epochs: ty.Optional[int]    
    mmd_train_result: ty.Optional[InterpretableMmdTrainResult]
    reg_parameter: ty.Optional[RegularizationParameter] = None
    execution_time_wallclock: ty.Optional[float] = None
    execution_time_cpu: ty.Optional[float] = None
    test_power_dev: ty.Optional[float] = None
    p_value_dev: ty.Optional[float] = None
    p_value_test: ty.Optional[float] = None
    



@dataclass
class ExecutionStatistics:
    __slots__ = ["regularization_parameters", "epochs", 
                 "execution_time_wall_clock", "execution_time_wall_cpu"]
    
    regularization_parameters: RegularizationParameter
    epochs: int
    execution_time_wall_clock: float
    execution_time_wall_cpu: float
    

@dataclass
class SelectionResult:                 
    regularization_parameters: ty.List[RegularizationParameter]  # regularization values obtained by the algorithm.
    selected_variables: ty.List[ty.Tuple[RegularizationParameter, ty.List[int]]]
    dict_regularization2model_parameter: ty.Optional[ty.Dict[RegularizationParameter, "BaseMmdEstimator"]]
    regularization_upper_searched: ty.Optional[RegularizationParameter] = None  # the maximum regularization parameter searched.
    regularization_lower_searched: ty.Optional[RegularizationParameter] = None  # the minimum regularization parameter searched.
    execution_statistics: ty.Optional[ty.List[ExecutionStatistics]] = None
    dict_regularization2optuna_return: ty.Optional[ty.Dict[RegularizationParameter, _DaskFunctionReturn]] = None

    @staticmethod
    def __key_sort(reg_parameter: RegularizationParameter) -> float:
        return reg_parameter.lambda_1 + reg_parameter.lambda_2
    
    def get_lower_bound_parameter(self) -> RegularizationParameter:
        min_param = min(self.regularization_parameters, key=self.__key_sort)
        return min_param

    def get_upper_bound_parameter(self) -> RegularizationParameter:
        min_param = max(self.regularization_parameters, key=self.__key_sort)
        return min_param
    
    def get_lower_bound_mmd_estimator(self) -> "BaseMmdEstimator":
        """I obtain an MMD-estimator at the lower bound of the regularization parameter.
        This implementation is from a study-issue-71 that fine-tuning works well.
        The fine-tuning is that you find lower and upper bound of regularization parameters by Optuna search,
        then you use the ARD weights at the lower bound for fine-tuning of the MMD-estimator.
        
        This idea makes required epochs fewer, hence shorter optimization time.
        """
        assert self.dict_regularization2model_parameter is not None
        parem_lower_bound = self.get_lower_bound_parameter()
        
        assert parem_lower_bound in self.dict_regularization2model_parameter
        
        return self.dict_regularization2model_parameter[parem_lower_bound]
    
    
@dataclass
class RegularizationSearchParameters:
    """Congiruation Dataclass for regularization parameter search.
    These parameters are used for `search_regularization_min_max.optuna_search.run_parameter_space_search`
    """
    search_strategy: str = 'optuna'

    n_search_iteration: int = 10
    max_concurrent_job: int = 2
    n_regularization_parameter: int = 6
    
    # if -1, then automatically determined.
    reg_parameter_search_lower_l1: float = 0.000001
    reg_parameter_search_lower_l2: float = 0.0

    reg_parameter_search_upper_l1: float = 2.0
    reg_parameter_search_upper_l2: float = 0.0
    
    
    backend: str = 'dask'  # single or dask
    path_optuna_study_db: ty.Optional[Path] = None

    def __post_init__(self):
        assert self.search_strategy in ('optuna', 'heuristic'), f"Invalid search strategy: {self.search_strategy}" 