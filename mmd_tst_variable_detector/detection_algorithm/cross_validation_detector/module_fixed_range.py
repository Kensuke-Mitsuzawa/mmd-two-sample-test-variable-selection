import typing as ty
import logging
import itertools
import random
import joblib
import copy
from pathlib import Path
from dataclasses import asdict

from dask.distributed import Client, LocalCluster

import torch
import pytorch_lightning as pl

from ...datasets import BaseDataset
from ...mmd_estimator.mmd_estimator import BaseMmdEstimator
from ...utils.post_process_logger import PostProcessLoggerHandler

from ..search_regularization_min_max.optuna_module.commons import SelectionResult, RegularizationSearchParameters
from ..search_regularization_min_max import (
    optuna_upper_lower_search,
    # hueristic_approach,
    hueristic_approach_dask
    )
from ..pytorch_lightning_trainer import PytorchLightningDefaultArguments 
from ..commons import (
    RegularizationParameter,
    InterpretableMmdTrainParameters
)

from .checkpoint_saver import CheckPointSaverStabilitySelection
from .commons import (
    CrossValidationTrainParameters,
    SubLearnerTrainingResult,
    RequestDistributedFunction,
    AggregationKey
)
from .module_data_sampling import DataSampling

from .module_utils import dask_worker_script
from ...logger_unit import handler


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


class SubModuleCrossValidationFixedRange(object):
    def __init__(self,
                 pytorch_trainer_config: PytorchLightningDefaultArguments,
                 training_parameter: CrossValidationTrainParameters,
                 estimator: BaseMmdEstimator,
                 post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
                 resume_checkpoint_saver : ty.Optional[CheckPointSaverStabilitySelection] = None,
                 cv_detection_experiment_name: ty.Optional[str] = None,
                 seed_root_random: int = 42,
                 dask_client: ty.Optional[Client] = None) -> None:
        self.dask_client = dask_client
        # -------------------------------------------------------
        # attributes
        self.training_dataset: BaseDataset
        self.validation_dataset: BaseDataset
        self.cv_detection_experiment_name = cv_detection_experiment_name
        self.data_sampling_module = DataSampling(seed_root_random=seed_root_random)
        # -------------------------------------------------------
        
        self.training_parameter = training_parameter
        self.resume_checkpoint_saver = resume_checkpoint_saver
        self.post_process_handler = post_process_handler
        self.pytorch_trainer_config = pytorch_trainer_config
        self.estimator = estimator
        self.seed_root_random = seed_root_random
    
    def __retrive_task_return_done(self, 
                                   sub_id_tuple: ty.List[ty.Tuple[RegularizationParameter, int]]
                                   ) -> ty.Tuple[ty.List[ty.Tuple[RegularizationParameter, int]], ty.List[SubLearnerTrainingResult]]:
        """Retriving tasks that are already done when the `resume_checkpoint_saver` is given. 
        And, it generates a list of request parameters.
        
        """
        # taking back from checkpoint
        if self.resume_checkpoint_saver is None:
            already_trained_sub_learners: ty.List[SubLearnerTrainingResult] = []
        else:
            already_trained_sub_learners: ty.List[SubLearnerTrainingResult] = self.resume_checkpoint_saver.load_checkpoint()
        # end if
        # __sub_id_tuple = itertools.product(reg_params, list(range(0, self.training_parameter.algorithm_parameter.n_subsampling)))

        already_trained_sub_learners_ids = [obj.job_id for obj in already_trained_sub_learners]
        # comment: `tuple_sub_id` is a tuple of (RegularizationParameter, subsampling-id).
        sub_id_tuple = [tuple_sub_id for tuple_sub_id in sub_id_tuple if tuple_sub_id not in already_trained_sub_learners_ids]

        return sub_id_tuple, already_trained_sub_learners
    
    def __log_post_process(self, seq_results_one_batch: ty.List[SubLearnerTrainingResult]):
        """Private API. Logging post-process results."""
        assert self.post_process_handler is not None, 'self.post_process_handler must not be None.'
        for __res in seq_results_one_batch:
            __run_name = __res.get_job_id_string()
            __loggers = self.post_process_handler.initialize_logger(run_name=__run_name, group_name=self.cv_detection_experiment_name)
            self.post_process_handler.log(loggers=__loggers, target_object=__res)
        # end for
    
    def __non_distributed_single_backend(self,
                                         seq_task_arguments: ty.List[RequestDistributedFunction]
                                         ) -> ty.List[SubLearnerTrainingResult]:
        """
        :return:
        """
        # seq_task_arguments = self.__generate_distributed_argument(sub_id_tuple)

        batch_n = self.training_parameter.distributed_parameter.job_batch_size
        seq_batch = [seq_task_arguments[i * batch_n:(i + 1) * batch_n] for i in range((len(seq_task_arguments) + batch_n - 1) // batch_n)]

        seq_results = []
        for on_job_batch in seq_batch:
            __seq_results_one_batch = []
            for args in on_job_batch:
                __seq_results_one_batch.append(dask_worker_script(args))
            # end for
            seq_results += __seq_results_one_batch
            # save opt results
            if self.resume_checkpoint_saver is not None:
                for sub_learner_result in __seq_results_one_batch:
                    self.resume_checkpoint_saver.save_checkpoint(sub_learner_result)
            # end if
            
            if self.post_process_handler is not None:
                logger.debug('logging post-process results...')
                self.__log_post_process(__seq_results_one_batch)
                logger.debug('logging Done')
            # end if
        # end for

        return seq_results
    
    def __distributed_dask_backend(self,
                                   seq_task_arguments: ty.List[RequestDistributedFunction]
                                #    sub_id_tuple: ty.List[ty.Tuple[RegularizationParameter, int]]
                                   ) -> ty.List[SubLearnerTrainingResult]:
        # seq_task_arguments = self.__generate_distributed_argument(sub_id_tuple)

        batch_n = self.training_parameter.distributed_parameter.job_batch_size
        seq_batch = [seq_task_arguments[i * batch_n:(i + 1) * batch_n] for i in range((len(seq_task_arguments) + batch_n - 1) // batch_n)]

        # client = Client(self.training_parameter.distributed_parameter.dask_scheduler_address)
        client = self.dask_client
        assert client is not None, 'Dask client is not given.'
        assert isinstance(client, Client), 'Dask client is not given.'
        seq_results = []
        for on_job_batch in seq_batch:
            task_queue = client.map(dask_worker_script, on_job_batch)
            __seq_results: ty.List[SubLearnerTrainingResult] = client.gather(task_queue)  # type: ignore
            seq_results += __seq_results

            # save opt results
            if self.resume_checkpoint_saver is not None:
                for sub_learner_result in __seq_results:
                    self.resume_checkpoint_saver.save_checkpoint(sub_learner_result)
            # end if

            if self.post_process_handler is not None:
                logger.debug('logging post-process results...')
                self.__log_post_process(__seq_results)
                logger.debug('logging Done')
            # end if
            
        return seq_results    
    
    def run_sub_learners_lambda_parameter(self,
                                          seq_job_function_parameter: ty.List[RequestDistributedFunction],
                                          trained_function_return_done: ty.List[SubLearnerTrainingResult]
                                          ) -> ty.List[SubLearnerTrainingResult]:
        """Public method.

        Run sub-learners with a regularization parameter. A controller of joblib or dask job scheduler.

        "job-id" consists of "LambdaParam-SubsamplingID".

        Parameters
        --------------        
        seq_job_function_parameter:
        trained_function_return_done:
        
        Returns
        --------------
        a list of `SubLearnerTrainingResult`
        """

        # if self.training_parameter.computation_backend == 'single':
        #     trained_function_return_done += self.__non_distributed_single_backend(seq_job_function_parameter)
        # elif self.training_parameter.computation_backend == 'joblib':
        #     trained_function_return_done += self.__distributed_joblib_backend(seq_job_function_parameter)
        # elif self.training_parameter.computation_backend == 'dask':
        #     trained_function_return_done += self.__distributed_dask_backend(seq_job_function_parameter)
        # else:
        #     raise NotImplementedError(f'No backend named {self.training_parameter.computation_backend}')
        # # end if
        if self.dask_client is None:
            trained_function_return_done += self.__non_distributed_single_backend(seq_job_function_parameter)
        else:
            trained_function_return_done += self.__distributed_dask_backend(seq_job_function_parameter)
        # end if
        
        return trained_function_return_done    
    
    def __generate_distributed_argument(self, 
                                        sub_id_tuple: ty.List[ty.Tuple[RegularizationParameter, int]]
                                        ) -> ty.List[RequestDistributedFunction]:
        """Setting training parameters for distributed computing.
        Dataset generation (data-splitting) is also done here.
        """
        lambda_parameter_unique = set([__t[0] for __t in sub_id_tuple])
        dict_parameter_type_ids = {_param_id: param for _param_id, param in enumerate(lambda_parameter_unique)}
        
        seq_task_datasets = self.data_sampling_module.get_datasets(
            seq_parameter_type_ids=list(dict_parameter_type_ids.keys()),
            n_sampling=self.training_parameter.algorithm_parameter.n_subsampling,
            training_dataset=self.training_dataset,
            validation_dataset=self.validation_dataset,
            sampling_strategy=self.training_parameter.algorithm_parameter.sampling_strategy,
            ratio_training_data=self.training_parameter.algorithm_parameter.ratio_subsampling,
        )
        
        seq_generated_task_request = []
        for __task_tuple in seq_task_datasets:
            _reg_parameter = dict_parameter_type_ids[__task_tuple.task_key.parameter_type_id]
            __task_id = AggregationKey(
                approach_regularization_parameter='fixed_range',
                trial_id_cross_validation=-1,  # I do not use this field.
                regularization=_reg_parameter,
                job_id=__task_tuple.task_key.data_splitting_id)
        
            __parameter: InterpretableMmdTrainParameters = copy.deepcopy(self.training_parameter.base_training_parameter)
            __parameter.regularization_parameter = _reg_parameter  # comment: s_id_tuple[0] is `RegularizationParameter`.
        
            # initializing the Pytorch Trainer.
            __sub_dir_name = f'{_reg_parameter.lambda_1}-{_reg_parameter.lambda_2}_{__task_tuple.task_key.data_splitting_id}'
                            
            if isinstance(self.pytorch_trainer_config.default_root_dir, Path):
                __path_default_root_dir = Path(self.pytorch_trainer_config.default_root_dir) / __sub_dir_name
            else:
                __path_default_root_dir = Path('/tmp/mmd-tst-variable-detector') / __sub_dir_name
            # end if
            __copy_pytorch_trainer_config: PytorchLightningDefaultArguments = copy.deepcopy(self.pytorch_trainer_config)
            __copy_pytorch_trainer_config.default_root_dir = __path_default_root_dir
            
            __trainer_lightning = pl.Trainer(**asdict(__copy_pytorch_trainer_config))
        
            __mmd_estimator = copy.deepcopy(self.estimator)
        
        
            __req = RequestDistributedFunction(
                task_id=__task_id,
                training_parameter=__parameter,
                dataset_train=__task_tuple.dataset_train,
                dataset_val=__task_tuple.dataset_dev,
                trainer_lightning=__trainer_lightning,
                mmd_estimator=__mmd_estimator,
                stability_algorithm_param=self.training_parameter.algorithm_parameter
                )
            seq_generated_task_request.append(__req)
        # end for
        return seq_generated_task_request
        
    def __generate_optimization_arguments(self,
                                          reg_params: ty.List[RegularizationParameter],
                                          reg_search_result: ty.Optional[SelectionResult] = None
                                          ) -> ty.Tuple[ty.List[RequestDistributedFunction], ty.List[SubLearnerTrainingResult]]:
        """Private method.
        Generating a list of request parameters.

        Parameters
        -----------------
        reg_search_result: ty.Optional[SelectionResult]
            An output from `run_parameter_space_search`. When it's given, I do fine-tuning.
            The fine-tuning is by `weights-of-lower-bound-reg`.
            That means, I set the ARD weights at the lower bound of the regularization parameter as the initial values.
        
        """
        sub_id_tuple = list(itertools.product(reg_params, list(range(0, self.training_parameter.algorithm_parameter.n_subsampling))))
        sub_id_tuple, already_trained_sub_learners = self.__retrive_task_return_done(sub_id_tuple)
        # ------------------------------------------------
        # NOTE: I have not confiemd the following code is necessary. I wrote it for making optimisation faster convergence.
        # getting weights at the lower bound of the regularization parameter.
        # if reg_search_result is not None:
        #     logger.info('reg_search_result is given. I use the weights at the lower bound of the regularization parameter.')
        #     assert reg_search_result.dict_regularization2model_parameter is not None
        #     __dict_reg2model = reg_search_result.dict_regularization2model_parameter.items()
        #     ## TODO L1 regularization parameter sorting only...
        #     t_mmd_model_lower = sorted(__dict_reg2model, key=lambda o: o[0][0])[0]
        #     mmd_model_lower = t_mmd_model_lower[1]
        #     if isinstance(mmd_model_lower, dict):
        #         lower_bound_weights = mmd_model_lower['kernel_obj.ard_weights']
        #     elif isinstance(mmd_model_lower, BaseMmdEstimator):
        #         lower_bound_weights = mmd_model_lower.kernel_obj.ard_weights
        #     else:
        #         raise ValueError(f'Unknown type of mmd_model_lower: {type(mmd_model_lower)}')
        #     # end if
        # else:
        #     lower_bound_weights = None
        # ------------------------------------------------
        # generating a list of request parameters.
        seq_task_arguments = self.__generate_distributed_argument(sub_id_tuple)
        
        return seq_task_arguments, already_trained_sub_learners    
        
    def main(self,
             training_dataset: BaseDataset,
             validation_dataset: ty.Optional[BaseDataset] = None
             ) -> ty.List[SubLearnerTrainingResult]:
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        
        # -------------------------------------------------------
        # # Dask client check or launching.
        # __client = self.__check_dask_client()
        __client = self.dask_client
        # -------------------------------------------------------
        # regularization parameter range
        if isinstance(self.training_parameter.algorithm_parameter.candidate_regularization_parameter, list):
            self.candidate_regularization_parameter = self.training_parameter.algorithm_parameter.candidate_regularization_parameter
        else:
            # Optuna based reg. parameter search.
            reg_search_config = self.training_parameter.algorithm_parameter.regularization_search_parameter
            if validation_dataset is None:
                dataset_whole = training_dataset
            else:    
                dataset_whole = training_dataset.merge_new_dataset(validation_dataset)
            # end if
            logger.debug('I am running search of lambda-parameter lower & upper bound...')
            _reg_lower = RegularizationParameter(
                lambda_1=reg_search_config.reg_parameter_search_lower_l1,
                lambda_2=reg_search_config.reg_parameter_search_lower_l2)
            _reg_upper = RegularizationParameter(
                lambda_1=reg_search_config.reg_parameter_search_upper_l1,
                lambda_2=reg_search_config.reg_parameter_search_upper_l2
            )
            
            if reg_search_config.search_strategy == 'optuna':
                # Optuna based search.
                reg_search_result = optuna_upper_lower_search.run_parameter_space_search(
                    dataset_train=dataset_whole,
                    mmd_estimator=self.estimator,
                    base_training_parameter=self.training_parameter.base_training_parameter,
                    pytorch_trainer_config=self.pytorch_trainer_config,
                    path_storage_backend_db=reg_search_config.path_optuna_study_db,
                    dask_client=__client,
                    post_process_handler=self.post_process_handler,
                    n_regularization_parameter=reg_search_config.n_regularization_parameter,
                    n_trials=reg_search_config.n_search_iteration,
                    concurrent_limit=reg_search_config.max_concurrent_job,
                    regularization_param_search_upper=_reg_upper,
                    regularization_param_search_lower=_reg_lower)
                self.candidate_regularization_parameter = reg_search_result.regularization_parameters
                logger.debug('Finished seahing of lambda-parameter lower & upper bound. Processing next step: CV MMD-Opt.')
            elif reg_search_config.search_strategy == 'heuristic':
                # Hueristic based search.
                reg_search_result = hueristic_approach_dask.run_parameter_space_search(
                    dataset=dataset_whole,
                    mmd_estimator=self.estimator,
                    training_parameter=self.training_parameter.base_training_parameter,
                    pytorch_trainer_config=self.pytorch_trainer_config,
                    n_regularization_parameter=reg_search_config.n_regularization_parameter,
                    max_try=reg_search_config.n_search_iteration,
                    post_process_handler=self.post_process_handler,
                    dask_client=__client,
                    n_concurrent_run=reg_search_config.max_concurrent_job,)
                self.candidate_regularization_parameter = reg_search_result.regularization_parameters
                logger.debug('Finished seahing of lambda-parameter lower & upper bound. Processing next step: CV MMD-Opt.')
            else:
                raise ValueError(f'Unknown search strategy: {reg_search_config.search_strategy}')
            # end if
        # end if 
        
        # running ARD weights optimization with sub-sampled dataset.
        
        reg_params = [RegularizationParameter(*__t) for __t in self.candidate_regularization_parameter]
    
    
        seq_task_arguments, trained_function_return_done = self.__generate_optimization_arguments(reg_params)
        
        if len(seq_task_arguments) == 0:
            # case: all tasks are already done. 
            seq_trained_variables = trained_function_return_done
            logger.info('All tasks are already done.')
        else:
            seq_trained_variables = self.run_sub_learners_lambda_parameter(
                seq_job_function_parameter=seq_task_arguments,
                trained_function_return_done=trained_function_return_done)
        # end if

        return seq_trained_variables