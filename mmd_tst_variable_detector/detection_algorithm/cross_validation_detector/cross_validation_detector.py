import typing
import copy
import logging
import functools

import dask
import dask.config

import dask.delayed
from dask.distributed import Client

from datetime import datetime
from dataclasses import asdict

import typing as ty

import torch
import numpy as np
import pytorch_lightning as pl

from ...exceptions import OptimizationException, SameDataException
from ...logger_unit import handler 
from ...datasets.base import BaseDataset
from ...mmd_estimator.mmd_estimator import BaseMmdEstimator
from ...utils.post_process_logger import PostProcessLoggerHandler

from ..interpretable_mmd_detector import InterpretableMmdDetector
from ..pytorch_lightning_trainer import PytorchLightningDefaultArguments 
from ..commons import (
    InterpretableMmdTrainResult, 
    RegularizationParameter,
)

from .checkpoint_saver import CheckPointSaverStabilitySelection
from .commons import (
    CrossValidationTrainParameters,
    CrossValidationAggregatedResult,
    CrossValidationAlgorithmParameter,
    CrossValidationTrainedParameter,
    ExecutionTimeStatistics,
)
from .module_aggregation import PostAggregatorMmdAGG
from .module_fixed_range import SubModuleCrossValidationFixedRange
from .module_param_search import SubModuleCrossValidationParameterSearching
from .module_utils import get_frequency_tensor

logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


# Independent functions called by Dask.

def post_ard_weight_optimization_hard(selected_indexes: ty.List[int],
                                      estimator: BaseMmdEstimator,
                                      training_parameter: CrossValidationTrainParameters,
                                      pytorch_trainer_config: PytorchLightningDefaultArguments,
                                      training_dataset: BaseDataset,
                                      validation_dataset: ty.Optional[BaseDataset]
                                      ) -> ty.Tuple[str, ty.Optional[InterpretableMmdTrainResult]]:
    """Re-run ARD optimization with the selected features.

    :param selected_indexes:
    :return:
    """
    assert len(selected_indexes) > 0, 'selected_indexes must have more than one index.'

    assert isinstance(selected_indexes, list) and all(isinstance(d, int) for d in selected_indexes)
    dimension_size: int = estimator.kernel_obj.ard_weights.shape[0]
    ignore_index = list(set(range(0, dimension_size)) - set(selected_indexes))  # type: ignore

    # region: replace initial_scale with 0.0
    initial_scale = torch.ones(estimator.kernel_obj.ard_weights.shape)
    for target_index in ignore_index:
        initial_scale[target_index] = 0.0
    # endregion

    estimator_copy = copy.deepcopy(estimator)
    estimator_copy.kernel_obj.ard_weights = torch.nn.Parameter(initial_scale)

    if training_dataset.is_dataset_on_ram():
        training_dataset = training_dataset.generate_dataset_on_ram()
    else:
        training_dataset = training_dataset
    # end if

    if validation_dataset is None:
        logger.debug('I use the training_dataset at the whole_dataset.')
        validation_dataset = training_dataset
    else:
        validation_dataset = validation_dataset
    # end if
    
    if validation_dataset.is_dataset_on_ram():
        validation_dataset = validation_dataset.generate_dataset_on_ram()
    else:
        validation_dataset = validation_dataset
    # end if

    sub_leaner = InterpretableMmdDetector(
        mmd_estimator=estimator_copy,
        training_parameter=training_parameter.base_training_parameter,
        dataset_train=training_dataset,
        dataset_validation=validation_dataset
    )

    try:
        # trainer_lightning = copy.deepcopy(self.trainer_lightning)
        trainer_lightning = pl.Trainer(**asdict(pytorch_trainer_config))
        trainer_lightning.fit(model=sub_leaner)
        training_log = sub_leaner.get_trained_variables()

        return 'post_ard_weight_optimization_hard', training_log
    except OptimizationException as e:
        logger.error(f'Error during MMD-Optimization after CV-aggregation. Exception message -> {e}')
        return 'post_ard_weight_optimization_hard', None
    

def post_ard_weight_optimization_soft(score_aggregated: CrossValidationAggregatedResult,
                                      estimator: BaseMmdEstimator,
                                      training_parameter: CrossValidationTrainParameters,
                                      pytorch_trainer_config: PytorchLightningDefaultArguments,
                                      training_dataset: BaseDataset,
                                      validation_dataset: ty.Optional[BaseDataset]
                                      ) -> ty.Tuple[str, ty.Optional[InterpretableMmdTrainResult]]:
    """Public method.

    Re-optimizing ARD weights. The initial values is normalized values of cross-validation results.

    Returns: `TrainingResult` or None. None when the initial MMD < 0.0 and Log mode is 1.0.
    """
    try:
        estimator_copy = copy.deepcopy(estimator)
        # use it as the initial value of ARD weights.
        assert isinstance(score_aggregated.array_s_hat, torch.Tensor)
        estimator_copy.kernel_obj.ard_weights = torch.nn.Parameter(score_aggregated.array_s_hat)
        
        if validation_dataset is None:
            logger.debug('I use the training_dataset at the whole_dataset.')
            validation_dataset = training_dataset
        else:
            validation_dataset = validation_dataset
        # end if
        
        if validation_dataset.is_dataset_on_ram():
            validation_dataset = validation_dataset.generate_dataset_on_ram()
        # end if
        
        if training_dataset.is_dataset_on_ram():
            training_dataset = training_dataset.generate_dataset_on_ram()
        else:
            training_dataset = training_dataset
        # end if

        sub_leaner = InterpretableMmdDetector(
            mmd_estimator=estimator_copy,
            training_parameter=training_parameter.base_training_parameter,
            dataset_train=training_dataset,
            dataset_validation=validation_dataset  # type: ignore
        )

        # trainer_lightning = copy.deepcopy(self.trainer_lightning)
        trainer_lightning = pl.Trainer(**asdict(pytorch_trainer_config))
        trainer_lightning.fit(model=sub_leaner)
        training_log = sub_leaner.get_trained_variables()
        
        return 'post_ard_weight_optimization_soft', training_log
    except OptimizationException as e:
        logger.error(f'Error during MMD-Optimization after CV-aggregation. Exception message -> {e}')
        return 'post_ard_weight_optimization_soft', None



class CrossValidationInterpretableVariableDetector(object):
    def __init__(self,
                 pytorch_trainer_config: PytorchLightningDefaultArguments,
                 training_parameter: CrossValidationTrainParameters,
                 estimator: BaseMmdEstimator,
                 post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
                 resume_checkpoint_saver : typing.Optional[CheckPointSaverStabilitySelection] = None,
                 training_dataset: ty.Optional[str] = None,
                 validation_dataset: ty.Optional[str] = None,
                 trainer_lightning: ty.Optional[pl.Trainer] = None,
                 seed_root_random: int = 42,
                 dask_client: ty.Optional[Client] = None):
        """
        
        Parameters
        -------------------
        
        pytorch_trainer_config: PytorchLightningDefaultArguments. Parameter for Pytorch Lightning Trainer.
        training_parameter: CrossValidationTrainParameters
        estimator: BaseMmdEstimator
        post_process_handler: ty.Optional[PostProcessLoggerHandler]
        resume_checkpoint_saver : typing.Optional[CheckPointSaverStabilitySelection]
            A Checkpoint handler saves the optimized results. The handler helps you resume the CV-Detection. 
        """
        self.dask_client = dask_client

        if trainer_lightning is not None:
            raise ValueError('trainer_lightning is deprecated. Give `pytorch_trainer_config` parameter object instead.')
        # end if
        
        self.pytorch_trainer_config = pytorch_trainer_config
        
        self.training_parameter = training_parameter
        self.estimator = estimator

        self.post_process_handler = post_process_handler
        self.resume_checkpoint_saver = resume_checkpoint_saver

        self.seq_trained_variables = []
        
        # unique of this run. used for logging mainly. 
        self.cv_detection_experiment_name = f'cv_detection-{datetime.now().isoformat()}'

        if training_dataset is not None:
            raise ValueError('training_dataset at __init__ is deprecated. Give it to `run_cv_detection` instead.')
        if validation_dataset is not None:
            raise ValueError('validation_dataset at __init__ is deprecated. Give it to `run_cv_detection` instead.')
        self.training_dataset: BaseDataset
        self.validation_dataset: ty.Optional[BaseDataset]
        
        self.candidate_regularization_parameter: ty.List[RegularizationParameter]
        
        if training_parameter.algorithm_parameter.approach_regularization_parameter == "fixed_range":
            self.sub_executor = SubModuleCrossValidationFixedRange(
                pytorch_trainer_config=self.pytorch_trainer_config,
                training_parameter=self.training_parameter,
                estimator=self.estimator,
                post_process_handler=self.post_process_handler,
                resume_checkpoint_saver=self.resume_checkpoint_saver,
                cv_detection_experiment_name=self.cv_detection_experiment_name,
                seed_root_random=seed_root_random,
                dask_client=dask_client,
            )
        elif training_parameter.algorithm_parameter.approach_regularization_parameter == "param_searching":
            self.sub_executor = SubModuleCrossValidationParameterSearching(
                pytorch_trainer_config=self.pytorch_trainer_config,
                training_parameter=self.training_parameter,
                estimator=self.estimator,
                post_process_handler=self.post_process_handler,
                resume_checkpoint_saver=self.resume_checkpoint_saver,
                cv_detection_experiment_name=self.cv_detection_experiment_name,
                dask_client=dask_client)
        else:
            raise ValueError(f"approach_regularization_parameter must be either of 'fixed_range' or 'param_searching'.")
        # end if
        
        self.post_aggregator = PostAggregatorMmdAGG(
            training_parameter=self.training_parameter,
            post_process_handler=self.post_process_handler,
            cv_detection_experiment_name=self.cv_detection_experiment_name)

    def run_stability_selection(self):
        """This function is just for keeping the version interchangeability.
        """
        raise ValueError('This API is deprecated. Use `run_cv_detection` instead.')

    def __train_selected_values(self, selected_indexes: ty.List[int]) -> InterpretableMmdTrainResult:
        """Re-run ARD optimization with the selected features.

        :param selected_indexes:
        :return:
        """

        assert len(selected_indexes) > 0, 'selected_indexes must have more than one index.'

        assert isinstance(selected_indexes, list) and all(isinstance(d, int) for d in selected_indexes)
        dimension_size: int = self.estimator.kernel_obj.ard_weights.shape[0]
        ignore_index = list(set(range(0, dimension_size)) - set(selected_indexes))  # type: ignore

        # region: replace initial_scale with 0.0
        initial_scale = torch.ones(self.estimator.kernel_obj.ard_weights.shape)
        for target_index in ignore_index:
            initial_scale[target_index] = 0.0
        # endregion

        estimator_copy = copy.deepcopy(self.estimator)
        estimator_copy.kernel_obj.ard_weights = torch.nn.Parameter(initial_scale)

        if self.training_dataset.is_dataset_on_ram():
            training_dataset = self.training_dataset.generate_dataset_on_ram()
        else:
            training_dataset = self.training_dataset
        # end if

        if self.validation_dataset is None:
            logger.debug('I use the training_dataset at the whole_dataset.')
            validation_dataset = self.training_dataset
        else:
            validation_dataset = self.validation_dataset
        # end if
        
        if validation_dataset.is_dataset_on_ram():
            validation_dataset = validation_dataset.generate_dataset_on_ram()
        else:
            validation_dataset = validation_dataset
        # end if

        sub_leaner = InterpretableMmdDetector(
            mmd_estimator=estimator_copy,
            training_parameter=self.training_parameter.base_training_parameter,
            dataset_train=training_dataset,
            dataset_validation=validation_dataset
        )

        # trainer_lightning = copy.deepcopy(self.trainer_lightning)
        trainer_lightning = pl.Trainer(**asdict(self.pytorch_trainer_config))
        trainer_lightning.fit(model=sub_leaner)
        training_log = sub_leaner.get_trained_variables()

        return training_log

    def train_post_mmd_estimators(self, 
                                  cv_aggregated: CrossValidationAggregatedResult,
                                  ) -> ty.Tuple[ty.Optional[InterpretableMmdTrainResult], ty.Optional[InterpretableMmdTrainResult]]:
        """Public method. Train post-ARD-optimization estimators."""
        assert cv_aggregated.stable_s_hat is not None and len(cv_aggregated.stable_s_hat) > 0
        
        # dask_client = self.__check_dask_client()
        
        if self.dask_client is None:
            __, estimator_hard = post_ard_weight_optimization_hard(
                selected_indexes=cv_aggregated.stable_s_hat,
                estimator=self.estimator,
                training_parameter=self.training_parameter,
                pytorch_trainer_config=self.pytorch_trainer_config,
                training_dataset=self.training_dataset,
                validation_dataset=self.validation_dataset)
            __, estimator_soft = post_ard_weight_optimization_soft(
                score_aggregated=cv_aggregated,
                estimator=self.estimator,
                training_parameter=self.training_parameter,
                pytorch_trainer_config=self.pytorch_trainer_config,
                training_dataset=self.training_dataset,
                validation_dataset=self.validation_dataset)
        else:            
            func_obj_train_hard = functools.partial(post_ard_weight_optimization_hard,
                selected_indexes=cv_aggregated.stable_s_hat,
                estimator=self.estimator,
                training_parameter=self.training_parameter,
                pytorch_trainer_config=self.pytorch_trainer_config,
                training_dataset=self.training_dataset,
                validation_dataset=self.validation_dataset
                )
            func_obj_train_soft = functools.partial(post_ard_weight_optimization_soft,
                score_aggregated=cv_aggregated,
                estimator=self.estimator,
                training_parameter=self.training_parameter,
                pytorch_trainer_config=self.pytorch_trainer_config,
                training_dataset=self.training_dataset,
                validation_dataset=self.validation_dataset
                )
            future_hard = self.dask_client.submit(func_obj_train_hard)
            future_soft = self.dask_client.submit(func_obj_train_soft)
            
            __, estimator_hard = future_hard.result()  # type: ignore
            __, estimator_soft = future_soft.result()  # type: ignore
        # end if
        
        if self.post_process_handler is not None:
            if estimator_hard is not None:
                __loggers_hard = self.post_process_handler.initialize_logger(run_name='post-estimator-hard', group_name=self.cv_detection_experiment_name)
                self.post_process_handler.log(loggers=__loggers_hard, target_object=estimator_hard)
            if estimator_soft is not None:
                __loggers_soft = self.post_process_handler.initialize_logger(run_name='post-estimator-soft', group_name=self.cv_detection_experiment_name)
                self.post_process_handler.log(loggers=__loggers_soft, target_object=estimator_soft)
            # end if
        # end if    
    
        return estimator_hard, estimator_soft

    def run_cv_detection(self,
                         training_dataset: BaseDataset,
                         validation_dataset: ty.Optional[BaseDataset] = None) -> CrossValidationTrainedParameter:
        """Public method.

        Interface method. Running a MMD optimization for ARD weights.
        Args:
            training_dataset: training dataset.
            validation_dataset: validation dataset. If give, I merge validation_dataset into training_dataset.
        Returns:
            `TrainedMmdParameters`
        """
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        
        if validation_dataset is not None:
            logger.info('validation_dataset is given. I merge it into training_dataset.')
        # end if
        
        seq_trained_variables = self.sub_executor.main(
            training_dataset=training_dataset, 
            validation_dataset=validation_dataset)
        
        # getting list of regularization parameters
        candidate_regularization_parameter = []
        for __train_obj in seq_trained_variables:
            candidate_regularization_parameter.append(tuple(__train_obj.job_id.regularization))
        # end for
        self.candidate_regularization_parameter = list(set(candidate_regularization_parameter))
            
        # -------------------------------------------------------
        # post-processing the results.
        seq_sub_estimator_container = [__res_obj.convert2SubEstimatorResultContainer() for __res_obj in seq_trained_variables]
        cv_aggregated, seq_agg_containers = self.post_aggregator.fit_transform(seq_sub_estimator_container)
        
        # -------------------------------------------------------
        # computing execution time statistics, epoch statistics
        seq_execution_time = np.array([__res_dask_func.execution_time_wallclock 
                                       for __res_dask_func in seq_trained_variables
                                       if __res_dask_func.execution_time_wallclock is not None])
        seq_epochs: ty.List[int] = [
            __res_dask_func.training_result.trajectory_record_training[-1].epoch 
            for __res_dask_func in seq_trained_variables 
            if __res_dask_func.training_result is not None]
        
        if len(seq_execution_time) == 0:
            logger.warning('No execution time/epoch is recorded.')
            exec_time_stats = ExecutionTimeStatistics(
                total_execution_time_wallclock=-1,
                avg_execution_time_wallclock=-1,
                std_execution_time_wallclock=-1,
                min_execution_time_wallclock=-1,
                max_execution_time_wallclock=-1,
                avg_epochs=-1,
                std_epochs=-1,
                min_epochs=-1,
                max_epochs=-1)
        else:
            __array_epochs = np.array(seq_epochs)
            exec_time_stats = ExecutionTimeStatistics(
                total_execution_time_wallclock=np.sum(seq_execution_time),
                avg_execution_time_wallclock=seq_execution_time.mean(),
                std_execution_time_wallclock=seq_execution_time.std(),
                min_execution_time_wallclock=np.min(seq_execution_time),
                max_execution_time_wallclock=np.max(seq_execution_time),
                avg_epochs=__array_epochs.mean(),
                std_epochs=__array_epochs.std(),
                min_epochs=np.min(__array_epochs),
                max_epochs=np.max(__array_epochs)
            )
        # end if
        
        # -------------------------------------------------------
        is_cv_result_none = cv_aggregated.stability_score_matrix is None or cv_aggregated.stable_s_hat is None
                
        if is_cv_result_none:
            ss_trained_parameter = CrossValidationTrainedParameter(
                regularization=self.candidate_regularization_parameter,
                stability_score_matrix=None,
                array_s_hat=None,
                stable_s_hat=[],
                variable_detection_postprocess_hard=None,
                execution_time_statistics=exec_time_stats,
                seq_aggregation_results=seq_agg_containers,
                training_parameters=self.training_parameter
            )
        elif cv_aggregated.stable_s_hat is not None and len(cv_aggregated.stable_s_hat) > 0:
            # training post-process estimators
            ss_trained_parameter_post_hard, ss_trained_parameter_post_soft = self.train_post_mmd_estimators(cv_aggregated)
            
            assert isinstance(cv_aggregated.array_s_hat, torch.Tensor)
            assert isinstance(cv_aggregated.stability_score_matrix, torch.Tensor)
            ss_trained_parameter = CrossValidationTrainedParameter(
                regularization=self.candidate_regularization_parameter,
                stability_score_matrix=cv_aggregated.stability_score_matrix.detach().cpu().numpy(),
                array_s_hat=cv_aggregated.array_s_hat.detach().cpu().numpy(),
                stable_s_hat=cv_aggregated.stable_s_hat,
                variable_detection_postprocess_hard=ss_trained_parameter_post_hard,
                variable_detection_postprocess_soft=ss_trained_parameter_post_soft,
                execution_time_statistics=exec_time_stats,
                seq_aggregation_results=seq_agg_containers,
                training_parameters=self.training_parameter,
                seq_sub_estimators=seq_sub_estimator_container)
        else:
            msg = 'No variable are detected. `cv_aggregated.stable_s_hat` is an empty list.'
            raise SameDataException(msg)
        # end if

        return ss_trained_parameter



# ---------------------------------------------------------------------
# class names for old package versions

StabilitySelectionVariableTrainer = CrossValidationInterpretableVariableDetector
StabilityScoreAggregatedResult = CrossValidationAggregatedResult
StabilitySelectionTrainedParameter = CrossValidationTrainedParameter
StabilitySelectionParameters = CrossValidationTrainParameters
StabilitySelectionAlgorithmParameter = CrossValidationAlgorithmParameter
