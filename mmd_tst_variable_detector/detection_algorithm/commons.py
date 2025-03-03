from dataclasses import dataclass, asdict
from pathlib import Path
import typing as ty
import uniplot
import matplotlib
import functools
from functools import partial
import logging

import matplotlib
import matplotlib.pyplot

import numpy as np
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..mmd_estimator import MmdValues, BaseMmdEstimator, ArgumentParameters
from ..datasets import BaseDataset
from ..logger_unit import handler


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


# ------------------------------------------------------------
# states objects used for training


@dataclass
class TrainingStatistics(object):
    global_step: int
    nan_frequency: int
    nan_ratio: ty.Optional[float] = None

    def __post_init__(self):
        self.nan_ratio = self.nan_frequency / self.global_step

@dataclass
class TrajectoryRecord:
    __slots__ = ('epoch', 'mmd', 'var', 'ratio', 'loss')
    
    epoch: int
    mmd: float
    var: float
    ratio: float
    loss: float


@dataclass
class DebugContainerNan(object):
    """Data container to record variables for debugging NAN issue."""

    epoch: int
    global_step: int
    mmd_values: MmdValues
    batch_xy: ty.Tuple[torch.Tensor, torch.Tensor]


class RegularizationParameter(ty.NamedTuple):
    # comment: this class should be with NamedTuple, for performance and maintainability.
    
    lambda_1: float
    lambda_2: float
    is_use_scientific_e: bool = True

    def __str__(self):
        if self.is_use_scientific_e:
            __l1 = "{:e}".format(self.lambda_1)
            __l2 = "{:e}".format(self.lambda_2)
        else:
            __l1 = str(self.lambda_1)
            __l2 = str(self.lambda_2)
        # end if
        
        return f"{__l1}-{__l2}"


@dataclass
class InterpretableMmdTrainParameters(object):
    """Parameter configurations for training interpretable MMD estimator.
    
    Parameters
    ------------
    batch_size: int
        Batch size for training. -1 means that the batch size is the same as the size of training data.
    regularization_parameter: RegularizationParameter
        Regularization parameter for MMD estimator.
    is_use_log: int
        0: no, 1: yes, -1: auto
    objective_function: str
        Either of following choices,
        1. 'ratio': ratio of MMD and variance
        2. 'mmd2': MMD^2
    frequency_epoch_trajectory_record: int
        Epoch frequency that the class keeps trajectory record. -1 does not save.
    lr_scheduler: ty.Optional[partial[ReduceLROnPlateau]]
        should be a partial function except `optimizer` argument.
    lr_scheduler_monitor_on: str
        Either of following choices,
        1. 'train_loss': use training loss for learning rate scheduler
        2. 'val_loss': use validation loss for learning rate scheduler
    optimizer_args: ty.Optional[ty.Dict]
        Arguments of `torch.optim.Adam`
    is_log_discrete_variables: bool
        Whether to log discrete variables. True then `detect_variables`.
    limit_steps_early_stop_nan: int
        Limit steps for early stopping when NAN is detected.
    limit_steps_early_stop_negative_mmd: int
        Limit steps for early stopping when negative MMD is detected.
    n_workers_train_dataloader: int
        Number of workers for training dataloader. Use > 1 when your dataset is with file backend.
    n_workers_validation_dataloader: int
        Number of workers for validation dataloader. Use > 1 when your dataset is with file backend.
    """
    batch_size: int = -1
    regularization_parameter: RegularizationParameter = RegularizationParameter(0.0, 0.0)
    is_use_log: int = 1  # -1: auto, 0: no, 1: yes
    objective_function: str = "ratio"  # ratio or mmd2
    frequency_epoch_trajectory_record: int = 100  # epoch frequency that the class keeps trajectory record. -1 does not save.
    lr_scheduler: ty.Optional[partial[ReduceLROnPlateau]] = None  # should be a partial function except `optimizer` argument.
    lr_scheduler_monitor_on: str = "train_loss"  # train_loss or val_loss
    optimizer_args: ty.Optional[ty.Dict] = None  # Arguments of `torch.optim.Adam`
    
    is_log_discrete_variables: bool = True
    
    limit_steps_early_stop_nan: int = 100
    limit_steps_early_stop_negative_mmd: int = 100
    
    n_workers_train_dataloader: int = 0
    n_workers_validation_dataloader: int = 0
    
    dataloader_persistent_workers: bool = False

    def __post_init__(self):
        if not isinstance(self.regularization_parameter, RegularizationParameter):
            if isinstance(self.regularization_parameter, (tuple, list)):
                self.regularization_parameter = RegularizationParameter(*self.regularization_parameter)  # type: ignore
            else:
                raise TypeError("regularization_parameter must be either a list or tuple.")
        # end if

        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, functools.partial)
        # end if
    

@dataclass
class InterpretableMmdTrainResult(object):
    ard_weights_kernel_k: torch.Tensor
    mmd_estimator: ty.Union[ty.Dict, "BaseMmdEstimator"]  # type: ignore
    training_stats: TrainingStatistics
    trajectory_record_training: ty.List[TrajectoryRecord]
    trajectory_record_validation: ty.List[TrajectoryRecord]
    training_parameter: ty.Optional[InterpretableMmdTrainParameters] = None
    training_configurations: ty.Optional[ty.Dict] = None
    mmd_estimator_hyperparameters: ty.Optional[ArgumentParameters] = None
    
    def to_dict(self):
        """Converting to a dictionary. Making possible to serialize.
        """
        if isinstance(self.mmd_estimator, pl.LightningModule):
            mmd_estimator = self.mmd_estimator.state_dict()
        else:
            mmd_estimator = self.mmd_estimator
        # end if
        
        trajectory_record_training = [asdict(__obj) for __obj in self.trajectory_record_training]
        trajectory_record_validation = [asdict(__obj) for __obj in self.trajectory_record_validation]
        
        dict_obj = dict(
            ard_weights_kernel_k=self.ard_weights_kernel_k,
            mmd_estimator=mmd_estimator,
            training_stats=asdict(self.training_stats),
            trajectory_record_training=trajectory_record_training,
            trajectory_record_validation=trajectory_record_validation,
            training_parameter=self.training_parameter,
            training_configurations=self.training_configurations,
            mmd_estimator_hyperparameters=self.mmd_estimator_hyperparameters)

        return dict_obj
        
    def plot_ard_weights(self, mode: str = "uniplot"):
        """Illustrating ARD weights.

        Parameters
        -------------
        uniplot, showing to a CUI console, matplotlib returns ax object.
        
        Returns
        -------------
        ax: matplotlib.pyplot.Axes
        """
        assert mode in ["uniplot", "matplotlib"]
        if mode == "uniplot":
            uniplot.plot(self.ard_weights_kernel_k)
        elif mode == "matplotlib":
            x_axis = range(len(self.ard_weights_kernel_k))
            ax = matplotlib.pyplot.bar(x_axis, self.ard_weights_kernel_k)
            return ax
        else:
            raise ValueError(f"mode must be either uniplot or matplotlib.")

    def plot_trajectory(self, mode: str = "uniplot", target: str = "validation", metric: str = "ratio"):
        assert mode in ["uniplot", "matplotlib"]
        assert target in ["training", "validation"]

        if target == "training":
            trajectory_record = self.trajectory_record_training
        elif target == "validation":
            trajectory_record = self.trajectory_record_validation
        else:
            raise ValueError(f"target must be either training or validation.")
        # end if

        target_metrics_values = [getattr(record, metric) for record in trajectory_record]

        if mode == "uniplot":
            uniplot.plot(
                target_metrics_values,
                title=f"Trajectory of {metric} ({target})",
            )
        elif mode == "matplotlib":
            x_axis = range(len(target_metrics_values))
            ax = matplotlib.pyplot.plot(x_axis, target_metrics_values)
            return ax
        else:
            raise ValueError(f"mode must be either uniplot or matplotlib.")


# # ------------------------------------------------------------
# # parameter objects for stability selection


# SAMPLING_STRATEGY = ('bootstrap', 'subsampling', 'cross-validation')
# STABILITY_SCORE_BASE = ('ard', 'variable')
# WEIGHTING_MODE = ('plane', 'p_value', 'test_power', 'p_value_filter', 'p_value_filter_test_power', 'p_value_min_test_power')


# @dataclass
# class DistributedComputingParameter:
#     """Configurations for distributed computing during Cross-Validation approach.
#     """
#     dask_scheduler_address: ty.Optional[str]
#     n_joblib: int = 2
#     joblib_backend: str = 'loky'  # recommend to use 'loky'. Most stable.
#     job_batch_size: int = 5
        
#     # when you wanna automatically launch Dask Cluster in the local,
#     # codebase uses the following parameters.
#     n_dask_workers: int = 4
#     n_threads_per_worker: int = 8

#     def __post_init__(self):
#         if self.dask_scheduler_address == "":
#             self.dask_scheduler_address = None

# @dataclass
# class RegularizationSearchParameters:
#     """Congiruation Dataclass for regularization parameter search.
#     These parameters are used for `search_regularization_min_max.optuna_search.run_parameter_space_search`
#     """
#     n_search_iteration: int = 10
#     max_concurrent_job: int = 2
#     n_regularization_parameter: int = 6
    
#     backend: str = 'dask'  # single or dask
#     path_optuna_study_db: ty.Optional[Path] = None
    

# @dataclass
# class CrossValidationAlgorithmParameter(object):
#     candidate_regularization_parameter: ty.Union[str, ty.List[RegularizationParameter]] = "auto"
#     regularization_search_parameter: RegularizationSearchParameters = RegularizationSearchParameters()
#     n_subsampling: int = 5
#     sampling_strategy: str = 'cross-validation'  # "subsampling", "bootstrap", "cross-validation"
#     ratio_subsampling: float = 0.8  # ratio to training data for cross-validation mode.
#     strategy_stability_score: str = 'mean'
#     threshold_stability_score: float = 0.1
#     weighting_mode: str = 'p_value_min_test_power'
#     stability_score_base: str = 'ard'  # ard or variable
#     ard_weight_minimum: float = 0.1
#     ard_weight_selection_strategy: str = 'hist_based'
#     is_normalize_agg_stability_score: bool = True
#     is_weight_stability_score: bool = True
#     permutation_test_metric: str = 'sliced_wasserstein'

#     def __post_init__(self):
#         assert self.sampling_strategy in SAMPLING_STRATEGY
#         assert self.stability_score_base in STABILITY_SCORE_BASE
#         assert self.weighting_mode in WEIGHTING_MODE
#         assert self.permutation_test_metric == 'sliced_wasserstein'

#         if isinstance(self.candidate_regularization_parameter, list):
#             assert len(self.candidate_regularization_parameter) > 0, "No parameter is given."
#             for __i, o in enumerate(self.candidate_regularization_parameter):
#                 if isinstance(o, tuple):
#                     self.candidate_regularization_parameter[__i] = RegularizationParameter(*o)
#         elif isinstance(self.candidate_regularization_parameter, str):
#             assert self.candidate_regularization_parameter == 'auto', "candidate_regularization_parameter must be either a list or auto"



# @dataclass
# class CrossValidationTrainParameters(object):
#     """
#     Parameters
#     ------------
#     computation_backend: str
#         Either of following choices,
#         1. 'single': Single machine
#         2. 'dask': Dask distributed computing
#         3. 'joblib': Joblib parallel computing
#     """
#     algorithm_parameter: CrossValidationAlgorithmParameter
#     base_training_parameter: InterpretableMmdTrainParameters
#     distributed_parameter: DistributedComputingParameter
#     computation_backend: str = 'dask'  # local or dask
#     dist_parameter: ty.Optional[DistributedComputingParameter] = None
#     wandb_logger_parameter = None    

#     def __post_init__(self):
#         assert self.computation_backend in ('single', 'dask', 'joblib')
        
#         if self.dist_parameter is not None:
#             logger.warning("dist_parameter is deprecated. Use distributed_parameter instead.")
#             self.distributed_parameter = self.dist_parameter
#         # end if
        
#         if self.wandb_logger_parameter is not None:
#             raise ValueError(
#                 "wandb_logger_parameter is deprecated. Delete it. Use post-process-logger instead."\
#                     "Adding `PostProcessLoggerHandler` to `CrossValidationInterpretableVariableDetector`")


# # ------------------------------------------------------------
# # parameter objects for stability selection

# @dataclass
# class RequestDistributedFunction:
#     __slots__ = ('task_id', 'training_parameter', 
#                  'dataset_train', 'dataset_val', 'trainer_lightning', 'mmd_estimator', 'stability_algorithm_param')
    
#     task_id: ty.Tuple[RegularizationParameter, int]
#     training_parameter: InterpretableMmdTrainParameters
#     dataset_train: BaseDataset
#     dataset_val: BaseDataset
#     trainer_lightning: pl.Trainer
#     mmd_estimator: "BaseMmdEstimator"  # type: ignore
#     stability_algorithm_param: CrossValidationAlgorithmParameter


# @dataclass
# class SubLearnerTrainingResult:
#     job_id: ty.Tuple[RegularizationParameter, int]
#     training_parameter: InterpretableMmdTrainParameters
#     training_result: ty.Optional[InterpretableMmdTrainResult]
#     p_value_selected: ty.Optional[float]
#     variable_detected: ty.Optional[ty.List[int]]
#     ard_weight_selected_binary: ty.Optional[torch.Tensor] = None
#     execution_time_wallclock: ty.Optional[float] = None
#     execution_time_cpu: ty.Optional[float] = None
#     epoch: ty.Optional[int] = None
    
#     def get_job_id_string(self, is_use_scientific_e: bool = True) -> str:
#         if is_use_scientific_e:
#             __l1 = "{:e}".format(self.job_id[0].lambda_1)
#             __l2 = "{:e}".format(self.job_id[0].lambda_2)
#         else:
#             __l1 = str(self.job_id[0].lambda_1)
#             __l2 = str(self.job_id[0].lambda_2)
#         # end if
        
#         return f"job-{__l1}-{__l2}-{self.job_id[1]}"
        

# ------------------------------------------------------------------
# Evaluation

@dataclass
class EvaluationVariableDetection:
    __slots__ = ('precision', 'recall', 'f1')
    
    precision: float
    recall: float
    f1: float


# ------------------------------------------------------------------
# For older version
TrainingParameters = InterpretableMmdTrainParameters
TrainingResult = InterpretableMmdTrainResult

