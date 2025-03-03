import typing as ty

from pathlib import Path
from dataclasses import dataclass

from frozendict import frozendict

import numpy as np
import torch
from pytorch_lightning.loggers.logger import Logger

from sklearn.linear_model import ARDRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR


AcceptableClass = ty.Union[SVR, ARDRegression, Ridge, LogisticRegression]


# ----------------------------------------------
# Public APIs


@dataclass
class TrainedResultRegressionBasedVariableDetector:
    regression_model: AcceptableClass
    p_value_soft_max: float
    p_value_hard_max: float
    weight_vector: ty.Optional[np.ndarray] = None
    selected_variable_indices: ty.Optional[ty.List[int]] = None
    seq_p_value_soft: ty.Optional[ty.List[float]] = None
    seq_p_value_hard: ty.Optional[ty.List[float]] = None
    execution_time_wallclock: ty.Optional[float] = None
    execution_time_cpu: ty.Optional[float] = None
    error_regression: ty.Optional[float] = None
    
    def to_dict(self):
        return {
            "regression_model": self.regression_model.__class__.__name__,
            "p_value_soft_max": self.p_value_soft_max,
            "p_value_hard_max": self.p_value_hard_max,
            "weight_vector": self.weight_vector,
            "selected_variable_indices": self.selected_variable_indices,
            "seq_p_value_soft": self.seq_p_value_soft,
            "seq_p_value_hard": self.seq_p_value_hard,
            "execution_time_wallclock": self.execution_time_wallclock, 
            "execution_time_cpu": self.execution_time_cpu,
            "error_regression": self.error_regression
        }


@dataclass
class CandidateModelContainer:
    model_candidate_id: str
    regression_models: ty.List[AcceptableClass]
    
    def __post_init__(self):
        self.pointer = 0
    
    def __len__(self):
        return len(self.regression_models)
    
    def __iter__(self):
        return self
    
    def __generate_key(self, param_container: AcceptableClass) -> str:
        return f"{param_container.__class__.__name__}_{param_container}_{self.pointer}"
    
    def __next__(self) -> ty.Tuple[str, AcceptableClass]:
        if self.pointer >= self.__len__():
            raise StopIteration

        __param = self.regression_models[self.pointer]
        __key = self.__generate_key(__param)
        self.pointer += 1
        return __key, __param
    
    def get_model_class_names(self) -> ty.List[str]:
        return [__param.__class__.__name__ for __param in self.regression_models]
    
    def get_model_keys(self) -> ty.List[str]:
        return [self.__generate_key(__param) for __param in self.regression_models]


@dataclass
class RegressionAlgorithmOneIndividualResult:
    model_obj: AcceptableClass
    selected_variables: ty.List[int]
    trained_weights: ty.Optional[ty.Union[np.ndarray, torch.Tensor]]
    seq_p_value_dev: ty.Optional[ty.List[float]]
    seq_p_value_test: ty.Optional[ty.List[float]] = None
    pl_loggers: ty.Optional[ty.List[Logger]] = None  # saving logger object. So, a user can do logging operation later.
    regression_model: ty.Optional[AcceptableClass] = None
    
    
@dataclass
class RegressionAlgorithmOneResult:
    __slots__ = ("selected_model", "trained_models")
    
    selected_model: ty.Optional[RegressionAlgorithmOneIndividualResult]
    trained_models: ty.List[RegressionAlgorithmOneIndividualResult]

# ----------------------------------------------
# aux clas definitions
# Private APIs

# class _FunctionRequestPayload(ty.NamedTuple):
#     # distributed computation parameter class. Private.
#     regression_model: AcceptableClass
#     dataset_training: BaseDataset
#     dataset_dev: BaseDataset
#     dataset_test: ty.Optional[BaseDataset] = None
#     set_permutation_test_runners: ty.Optional[ty.List[PermutationTest]] = None
#     variable_detection_method: str = "hist_based"
#     path_work_dir: ty.Optional[Path] = None
#     pl_loggers: ty.Optional[ty.List[Logger]] = None
#     task_id: ty.Optional[str] = None
#     model_type_id: ty.Optional[str] = None
#     batch_size: int = -1
#     tst_approach: str = "hard"  # sort or hard
    
#     def to_dict(self):
#         return {
#             "regression_model": self.regression_model.__class__.__name__,
#             "dataset_training": self.dataset_training.__class__.__name__,
#             "dataset_dev": self.dataset_dev.__class__.__name__,
#             "dataset_test": self.dataset_test.__class__.__name__ if self.dataset_test is not None else None,
#             "variable_detection_method": self.variable_detection_method,
#             "path_work_dir": self.path_work_dir,
#             "task_id": self.task_id,
#             "model_type_id": self.model_type_id,
#             "batch_size": self.batch_size,
#             "tst_approach": self.tst_approach
#         }


# class _FunctionReturn(ty.NamedTuple):
#     # return of distributed computation. Private.
#     request: _FunctionRequestPayload
#     trained_model: AcceptableClass
#     weights: np.ndarray
#     indices_detected: ty.List[int]
#     seq_p_value_dev: ty.Optional[ty.List[float]] = None
#     seq_p_value_test: ty.Optional[ty.List[float]] = None
#     is_success: ty.Optional[bool] = False
#     wallclock_execution_time: ty.Optional[float] = None
#     cpu_time_execution_time: ty.Optional[float] = None
    
#     def to_dict(self):
#         return {
#             "request": self.request.to_dict(),
#             "trained_model": self.trained_model,
#             "weights": self.weights,
#             "indices_detected": self.indices_detected,
#             "seq_p_value_dev": self.seq_p_value_dev,
#             "seq_p_value_test": self.seq_p_value_test,
#             "is_success": self.is_success,
#             "wallclock_execution_time": self.wallclock_execution_time,
#             "cpu_time_execution_time": self.cpu_time_execution_time
#         }


# ----------------------------------------------

from optuna.trial import Trial


@dataclass
class BaseSearchParameter:
    class_name: str
    params: ty.Optional[frozendict] = None


@dataclass
class SearchParameterSVR(BaseSearchParameter):
    class_name: str = "SVR"
    kernel = "linear"
    degree = (1, 5)
    coef0 = (0.0, 1.0)
    C = (0.0, 3.0) 
    # params: frozendict = frozendict({
    #     'kernel': "linear",
    #     'degree': (1, 5),
    #     'coef0': (0.0, 1.0),
    #     'C': (0.0, 3.0)
    # })


@dataclass
class SearchParameterLogisticRegression(BaseSearchParameter):
    class_name: str = "LogisticRegression"
    # penalty = [None, 'l1', 'l2', 'elasticnet']  # elasticnet is enough to use.
    penalty = ['elasticnet']
    C = (0.0, 5.0)
    l1_ratio = (0.0, 1.0)
    
    # params: frozendict = frozendict({
    #     'penalty': ['l1', 'l2', 'elasticnet'],
    #     'C': (0.0, 5.0)
    # })
    

@dataclass
class SearchParameterRidge(BaseSearchParameter):
    class_name: str = "Ridge"
    alpha = (0.0, 5.0)
    # params: frozendict = frozendict({
    #     'alpha': (0.0, 5.0)
    # })
    

@dataclass
class SearchParameterARDRegression(BaseSearchParameter):
    class_name: str = "ARDRegression"
    # params: frozendict = frozendict({
    #     'alpha_1': (0.0, 5.0),
    #     'alpha_2': (0.0, 5.0),
    #     'lambda_1': (0.0, 5.0),
    #     'lambda_2': (0.0, 5.0)
    # })
    alpha_1: ty.Tuple[float, float] = (0.0, 5.0)
    alpha_2: ty.Tuple[float, float] = (0.0, 5.0)
    lambda_1: ty.Tuple[float, float] = (0.0, 5.0)
    lambda_2: ty.Tuple[float, float] = (0.0, 5.0)

