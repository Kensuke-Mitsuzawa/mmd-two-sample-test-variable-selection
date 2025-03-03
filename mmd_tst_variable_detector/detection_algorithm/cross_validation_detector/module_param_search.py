import typing as ty
import logging
import itertools
import random
import joblib
import copy
from pathlib import Path
from dataclasses import asdict

from dask.distributed import Client, LocalCluster

import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.model_selection import KFold

from ...datasets import BaseDataset
from ...mmd_estimator.mmd_estimator import BaseMmdEstimator
from ...utils.post_process_logger import PostProcessLoggerHandler


from ..search_regularization_min_max.optuna_module.commons import SelectionResult, RegularizationSearchParameters
from ..search_regularization_min_max import optuna_opt_parameter_search
from ..pytorch_lightning_trainer import PytorchLightningDefaultArguments 
from ..commons import (
    RegularizationParameter,
    InterpretableMmdTrainParameters
)

from .commons import CrossValidationTrainParameters
from .checkpoint_saver import CheckPointSaverStabilitySelection
from .commons import (
    AggregationKey,
    CrossValidationTrainParameters,
    SubLearnerTrainingResult,
    RequestDistributedFunction,
)


from ...logger_unit import handler


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


# ------------------------------------------------------------
# Internal Variable


class _TaskTuple(ty.NamedTuple):
    task_key: AggregationKey
    dataset_train: BaseDataset
    dataset_dev: BaseDataset
    dataset_test: ty.Optional[BaseDataset]



class SubModuleCrossValidationParameterSearching(object):
    def __init__(self,
                 pytorch_trainer_config: PytorchLightningDefaultArguments,
                 training_parameter: CrossValidationTrainParameters,
                 estimator: BaseMmdEstimator,
                 post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
                 resume_checkpoint_saver : ty.Optional[CheckPointSaverStabilitySelection] = None,
                 cv_detection_experiment_name: ty.Optional[str] = None,
                 path_work_dir: ty.Optional[Path] = None,
                 seed_root_random: int = 42) -> None:
        # -------------------------------------------------------
        # attributes
        self.training_dataset: BaseDataset
        self.validation_dataset: BaseDataset
        self.cv_detection_experiment_name = cv_detection_experiment_name
        # -------------------------------------------------------
        
        self.training_parameter = training_parameter
        self.post_process_handler = post_process_handler
        self.pytorch_trainer_config = pytorch_trainer_config
        self.estimator = estimator
        
        self.path_work_dir = path_work_dir
        self.seed_root_random = seed_root_random
        
        # TODO resume_checkpoint_saver not used currently.
        # self.resume_checkpoint_saver = resume_checkpoint_saver
        
    def __check_dask_client(self) -> ty.Optional[Client]:
        """Private method. 
        Checking Dask client. If it is not given, I launch a local cluster.
        """
        __client = None
        if self.training_parameter.computation_backend == 'dask':
            if self.training_parameter.distributed_parameter.dask_scheduler_address is None:
                logger.debug('Dask scheduler is not given. I launch a local cluster.')
                __dask_cluster = LocalCluster(
                    n_workers=self.training_parameter.distributed_parameter.n_dask_workers,
                    threads_per_worker=self.training_parameter.distributed_parameter.n_threads_per_worker)
                __client = __dask_cluster.get_client()
                logger.debug(f'Dask scheduler address: {__dask_cluster.scheduler_address}')
                self.training_parameter.distributed_parameter.dask_scheduler_address = __dask_cluster.scheduler_address
            else:            
                try:
                    __client = Client(self.training_parameter.distributed_parameter.dask_scheduler_address)
                    logger.info(f'Connected to Dask scheduler: {__client}')
                except OSError:
                    logger.warning('Dask scheduler is not found. I run the computation in a single machine.')
                    self.training_parameter.computation_backend = 'single'
                # end try
            # end if
        else:
            __client = None
        # end if
        return __client        
    
    # def __retrive_task_return_done(self, 
    #                                sub_id_tuple: ty.List[ty.Tuple[RegularizationParameter, int]]
    #                                ) -> ty.Tuple[ty.List[ty.Tuple[RegularizationParameter, int]], ty.List[SubLearnerTrainingResult]]:
    #     """Retriving tasks that are already done when the `resume_checkpoint_saver` is given. 
    #     And, it generates a list of request parameters.
        
    #     """
    #     # taking back from checkpoint
    #     if self.resume_checkpoint_saver is None:
    #         already_trained_sub_learners: ty.List[SubLearnerTrainingResult] = []
    #     else:
    #         already_trained_sub_learners: ty.List[SubLearnerTrainingResult] = self.resume_checkpoint_saver.load_checkpoint()
    #     # end if
    #     # __sub_id_tuple = itertools.product(reg_params, list(range(0, self.training_parameter.algorithm_parameter.n_subsampling)))

    #     already_trained_sub_learners_ids = [obj.job_id for obj in already_trained_sub_learners]
    #     # comment: `tuple_sub_id` is a tuple of (RegularizationParameter, subsampling-id).
    #     sub_id_tuple = [tuple_sub_id for tuple_sub_id in sub_id_tuple if tuple_sub_id not in already_trained_sub_learners_ids]

    #     return sub_id_tuple, already_trained_sub_learners
    
    def __generate_cross_validation_datasets(self,
                                             training_dataset: BaseDataset,
                                             validation_dataset: ty.Optional[BaseDataset],
                                             test_dataset: ty.Optional[BaseDataset] = None
                                             ) -> ty.List[_TaskTuple]:
        """Generating parameters for cross-validation training.
        
        Args
        ----
        training_dataset: BaseDataset
            Training dataset.
        validation_dataset: ty.Optional[BaseDataset]
            Validation dataset, but not used as the validation dataset.
            If given, I merge it into the training dataset, and create a new dataset.
        test_dataset: ty.Optional[BaseDataset]
            Test dataset.
            If given, p-value of test data is calculated.
        """
        # mergin datasets if validation dataset is given.
        if validation_dataset is not None:
            dataset_whole = training_dataset.merge_new_dataset(validation_dataset)
        else:
            dataset_whole = training_dataset
        # end if
        
        sampling_strategy = self.training_parameter.algorithm_parameter.sampling_strategy
        seq_sample_ids = list(range(0, len(dataset_whole)))

        local_random_gen = random.Random(self.seed_root_random)
        random_seed_ids = [local_random_gen.randint(0, 999) for _ in range(0, self.training_parameter.algorithm_parameter.n_subsampling)]
        assert len(random_seed_ids) == self.training_parameter.algorithm_parameter.n_subsampling

        seq_task_parameters: ty.List[_TaskTuple] = []
        
        if sampling_strategy == 'cross-validation':
            _ratio_sampling = self.training_parameter.algorithm_parameter.ratio_subsampling
            _n_subsampling = self.training_parameter.algorithm_parameter.n_subsampling
            for __n_cv_trial in range(0, _n_subsampling):
                __n_sample_train = int(len(dataset_whole) * _ratio_sampling)
                __local_sampling_random_gen = random.Random(random_seed_ids[__n_cv_trial])                
                sample_ids_train = __local_sampling_random_gen.sample(range(len(dataset_whole)), k=__n_sample_train)
                sample_ids_val = list(set(range(len(dataset_whole))) - set(sample_ids_train))
                
                __, new_dataset_train = dataset_whole.get_subsample_dataset(sample_ids=sample_ids_train)
                __, new_dataset_val = dataset_whole.get_subsample_dataset(sample_ids=sample_ids_val)
                _new_dataset_train = new_dataset_train.copy_dataset()
                _new_dataset_val = new_dataset_val.copy_dataset()
                __task_key = AggregationKey(
                    approach_regularization_parameter='param_searching',
                    trial_id_cross_validation=__n_cv_trial,
                    regularization=None,
                    job_id=None)
                seq_task_parameters.append(_TaskTuple(
                    task_key=__task_key,
                    dataset_train=_new_dataset_train,
                    dataset_dev=_new_dataset_val,
                    dataset_test=test_dataset))
            # end for
        elif sampling_strategy == 'fold-cross-validation':
            _k_fold = self.training_parameter.algorithm_parameter.n_subsampling
            _k_fold_splitter = KFold(_k_fold, shuffle=False)  # shuffle off for the better re-producibility.
            for i, (train_index, test_index) in enumerate(_k_fold_splitter.split(np.array(seq_sample_ids))):
                assert train_index is not None and test_index is not None
                __, _train_dataset = dataset_whole.get_subsample_dataset(sample_ids=train_index.tolist())
                __, _dev_dataset = dataset_whole.get_subsample_dataset(sample_ids=test_index.tolist())

                _new_dataset_train = _train_dataset.copy_dataset()
                _new_dataset_val = _dev_dataset.copy_dataset()
                
                __task_key = AggregationKey(
                    approach_regularization_parameter='param_searching',
                    trial_id_cross_validation=i,
                    regularization=None,
                    job_id=None)
                seq_task_parameters.append(_TaskTuple(
                    task_key=__task_key,
                    dataset_train=_new_dataset_train,
                    dataset_dev=_new_dataset_val,
                    dataset_test=test_dataset))
            # for end
        else:
            raise NotImplementedError(f'{sampling_strategy} is not implemented.')
        # end if
        return seq_task_parameters                
    
    def __execute_search_parameter(self, 
                                   cv_parameters: ty.List[_TaskTuple],
                                   dask_client: ty.Optional[Client] = None
                                   ) -> ty.List[SubLearnerTrainingResult]:
        seq_sublearner_result: ty.List[SubLearnerTrainingResult] = []
        # this module can execute parameter search within the same dataset.
        # for the moment, it's okay.
        for __cv_task_parameter in cv_parameters:
            _reg_lower = RegularizationParameter(
                lambda_1=self.training_parameter.algorithm_parameter.regularization_search_parameter.reg_parameter_search_lower_l1,
                lambda_2=self.training_parameter.algorithm_parameter.regularization_search_parameter.reg_parameter_search_lower_l2)
            _reg_upper = RegularizationParameter(
                lambda_1=self.training_parameter.algorithm_parameter.regularization_search_parameter.reg_parameter_search_upper_l1,
                lambda_2=self.training_parameter.algorithm_parameter.regularization_search_parameter.reg_parameter_search_upper_l2
            )
            
            selection_result = optuna_opt_parameter_search.main(
                dataset_train=__cv_task_parameter.dataset_train,
                dataset_dev=__cv_task_parameter.dataset_dev,
                dataset_test=__cv_task_parameter.dataset_test,
                mmd_estimator=self.estimator,
                base_training_parameter=self.training_parameter.base_training_parameter,
                pytorch_trainer_config=self.pytorch_trainer_config,
                path_storage_backend_db=self.training_parameter.algorithm_parameter.regularization_search_parameter.path_optuna_study_db,
                dask_client=dask_client,
                post_process_handler=self.post_process_handler,
                n_trials=self.training_parameter.algorithm_parameter.regularization_search_parameter.n_search_iteration,
                concurrent_limit=self.training_parameter.algorithm_parameter.regularization_search_parameter.max_concurrent_job,
                regularization_param_search_upper=_reg_upper,
                regularization_param_search_lower=_reg_lower,
                path_work_dir=self.path_work_dir,
                n_permutation_test=self.training_parameter.algorithm_parameter.n_permutation_test,)
            
            # transferring objects from `SelectionResult` to `SubLearnerTrainingResult`
            _dict_regularization2optuna_return = selection_result.dict_regularization2optuna_return
            assert _dict_regularization2optuna_return is not None
            
            for _reg_param, _dask_func_out in _dict_regularization2optuna_return.items():                
                _job_id_obj = AggregationKey(
                    approach_regularization_parameter=__cv_task_parameter.task_key.approach_regularization_parameter,
                    trial_id_cross_validation=__cv_task_parameter.task_key.trial_id_cross_validation,
                    regularization=_reg_param,
                    job_id=_dask_func_out.trial.number)


                if _dask_func_out.mmd_train_result is None:
                    training_parameter = None
                    _ard_weight_selected_binary = None
                else:
                    training_parameter = _dask_func_out.mmd_train_result.training_parameter
                    if _dask_func_out.mmd_train_result.ard_weights_kernel_k is None:
                        _ard_weight_selected_binary = None
                    else:
                        _ard_weight_selected_binary = _dask_func_out.mmd_train_result.ard_weights_kernel_k
                    # end if                    
                # end if
                
                _dask_sub_learner_res = SubLearnerTrainingResult(
                    job_id=_job_id_obj,
                    training_parameter=training_parameter,
                    training_result=_dask_func_out.mmd_train_result,
                    p_value_selected=_dask_func_out.p_value_dev,
                    variable_detected=_dask_func_out.selected_variables,
                    ard_weight_selected_binary=_ard_weight_selected_binary,
                    execution_time_wallclock=-1,
                    execution_time_cpu=-1,
                    epoch=_dask_func_out.epochs)
                seq_sublearner_result.append(_dask_sub_learner_res)
            # end for
        # end for
        return seq_sublearner_result
         
    def main(self,
             training_dataset: BaseDataset,
             validation_dataset: ty.Optional[BaseDataset] = None,
             test_dataset: ty.Optional[BaseDataset] = None
             ) -> ty.List[SubLearnerTrainingResult]:
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        
        # splitting into CV datasets
        # returning object like: [(AggregationKey, training-dataset, validation-dataset), ...]
        seq_parameters_cv = self.__generate_cross_validation_datasets(
            training_dataset=self.training_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=test_dataset)
                
        # -------------------------------------------------------
        # Dask client check or launching.
        __client = self.__check_dask_client()
        seq_sub_learner_obj = self.__execute_search_parameter(seq_parameters_cv, __client)
        
        return seq_sub_learner_obj
