from collections.abc import Iterable
import logging
import functools
import typing as ty
from pathlib import Path
from tempfile import mkdtemp
from copy import deepcopy
from dataclasses import asdict, dataclass
import time
import timeit
import traceback

import optuna

import pytorch_lightning as pl

import torch
import numpy as np

from ....datasets import BaseDataset
from ....datasets.file_onetime_load_backend_static_dataset import FileBackendOneTimeLoadStaticDataset
from ....utils.variable_detection import detect_variables
from ....utils import PostProcessLoggerHandler

from ....mmd_estimator.mmd_estimator import BaseMmdEstimator
from ...interpretable_mmd_detector import InterpretableMmdDetector
from ...pytorch_lightning_trainer import PytorchLightningDefaultArguments
from ...commons import (
    RegularizationParameter, 
    InterpretableMmdTrainParameters, 
    InterpretableMmdTrainResult)
from .commons import ExecutionStatistics, _DaskFunctionReturn
from ...utils.permutation_tests import permutation_tests

from ....exceptions import OptimizationException, ParameterSearchException


logger = logging.getLogger(f'{__package__}.{__name__}')




def __function_wrapper_optuna(dict_l1_l2_parameter: ty.Dict[str, ty.Union[int, float]],
                              pytorch_trainer_config: PytorchLightningDefaultArguments,
                              dataset_train: BaseDataset,
                              dataset_test: BaseDataset,
                              mmd_estimator: BaseMmdEstimator,
                              training_parameter: InterpretableMmdTrainParameters,
                              ) -> ty.Tuple[ty.Optional[InterpretableMmdTrainResult], float]:
   """Searching the upper-bound of the lambda parameter.
   The objective function is the smallest number of detected variables."""
   # take out the reg parameter.   
   
   l1 = dict_l1_l2_parameter.get('l1', None)
   l2 = dict_l1_l2_parameter.get('l2', None)
   trial_id = dict_l1_l2_parameter.get('id', None)
   assert l1 is not None, f'l1 must not be None.'
   assert l2 is not None, f'l2 must not be None.'
   assert trial_id is not None, f'trial_id must not be None.'
      
   new_training_param = deepcopy(training_parameter)
   new_training_param.regularization_parameter = RegularizationParameter(
        lambda_1=l1, lambda_2=l2)
   
   logger.debug(f'lambda_1 = {l1}, lambda_2 = {l2}')
   
   try:
       variable_trainer = InterpretableMmdDetector(
            mmd_estimator=deepcopy(mmd_estimator),
            training_parameter=new_training_param,
            dataset_train=dataset_train,
            dataset_validation=dataset_test)              
       trainer_pl = pl.Trainer(**asdict(pytorch_trainer_config))
    #    variable_trainer = torch.compile(variable_trainer)
       trainer_pl.fit(variable_trainer)
   except OptimizationException as e:
         logger.warning(f'OptimizationException: {e}.'
                        'Either inproper regularization parameter or the data-pair (x, y) are same.')
         return None, trial_id
   except Exception as e:
       logger.error(f'Exception: {e}. Traceback -> {traceback.format_exc()}')
       return None, trial_id
   else:   
       # getting variables
       __trained_result = variable_trainer.get_trained_variables()

       return __trained_result, trial_id
    # end if



def __get_invalid_objective_value(trial: optuna.Trial, 
                                  objective_function: str,
                                  detection_result: ty.Optional[InterpretableMmdTrainResult]
                                  ) -> float:
    if detection_result is None:
        if objective_function == 'ratio_minimum':
            return 1.0
        elif objective_function == 'ratio_maximum':
            return -1.0
        elif objective_function == 'testpower_pvalue':
            return 1.0
        else:
            raise NotImplementedError(f'objective_function = {objective_function}')
        # end if
    # end if

    if objective_function in ('ratio_minimum', 'ratio_maximum'):
        # a value of trial study direction
        # NOT_SET = 0, MINIMIZE = 1, MAXIMIZE = 2
        direction_value = trial.study.direction
        if direction_value == 1:
            invalid_obj_value = 1.0
        elif direction_value == 2:
            invalid_obj_value = -1.0
        else:
            raise NotImplementedError(f'direction_value = {direction_value}')
        # end if
    elif objective_function == 'testpower_pvalue':
        # minimizing (-1 * (1.0 - p-value) * test-power).
        # so, penalty is a big value.
        invalid_obj_value = 1.0
    else:
        raise NotImplementedError(f'objective_function = {objective_function}')
    # end if
    return invalid_obj_value
    


def __get_optuna_objective_value(trial: optuna.Trial, 
                                 objective_function: str,
                                 detection_result: ty.Optional[InterpretableMmdTrainResult],
                                 p_value_dev: ty.Optional[float],
                                 variables_without_regularization: ty.Optional[ty.List[int]] = None,
                                 ) -> float:
    """Definition of Optuna objective function."""
    assert objective_function in ('ratio_minimum', 'ratio_maximum', 'testpower_pvalue')
    
    invalid_obj_value = __get_invalid_objective_value(trial, objective_function, detection_result)
    if detection_result is None:
        return invalid_obj_value
    # end if
    __variables = detect_variables(variable_weights=detection_result.ard_weights_kernel_k)
        
    
    if objective_function in ('ratio_minimum', 'ratio_maximum'):
        # --------------------------------------------------
        # ratio based objective function
        ratio_variables = len(__variables) / len(detection_result.ard_weights_kernel_k)
        return ratio_variables
        
    elif objective_function == 'testpower_pvalue':
        assert p_value_dev is not None
        # minimizing (-1 * (1.0 - p-value) * test-power)
        # p-value is the probability of rejecting the null hypothesis when it is true.
        # test-power is the probability of rejecting the null hypothesis when it is false.
        if variables_without_regularization is not None:
            if len(__variables) > len(variables_without_regularization):
                # When too many detected-variables, then invalid objective value. I return a big value.
                return 100.0
            # end if
        # end if
        testpower_dev = detection_result.trajectory_record_validation[-1].ratio
        value_obj = -1 * (1.0 - p_value_dev) * testpower_dev
        return value_obj
    else:
        raise NotImplementedError(f'objective_function = {objective_function}')
    

def func_dask_weapper_function_optuna(seq_trial: ty.List[optuna.Trial],
                                      objective_function: str,
                                      path_work_dir: Path,
                                      dataset_train: BaseDataset,
                                      dataset_dev: BaseDataset,
                                      pytorch_trainer_config: PytorchLightningDefaultArguments,
                                      search_lower_l1: float,
                                      search_upper_l1: float,
                                      search_lower_l2: float,
                                      search_upper_l2: float,
                                      mmd_estimator: BaseMmdEstimator,
                                      training_parameter: InterpretableMmdTrainParameters,
                                      dask_client: ty.Optional[ty.Any] = None,
                                      dataset_test: ty.Optional[BaseDataset] = None,
                                      ratio_variable_current_max: ty.Optional[float] = None,
                                      test_distance_functions: ty.Tuple[str, ...] = ('sliced_wasserstein',),
                                      variables_without_regularization: ty.Optional[ty.List[int]] = None,
                                      n_permutation_test: int = 500
                                      ) -> ty.List[_DaskFunctionReturn]:
    """This function is called by Dask. Optuna objective function is also in this function.
    """
    # start counting execution time
    start_cpu_time = time.process_time()
    start_wall_time = timeit.default_timer()
    
    if dataset_train.is_dataset_on_ram():
        dataset_train = dataset_train.generate_dataset_on_ram()
    # end if
    if dataset_dev.is_dataset_on_ram():
        dataset_dev = dataset_dev.generate_dataset_on_ram()
    # end if
    if dataset_test is not None and dataset_test.is_dataset_on_ram():
        dataset_test = dataset_test.generate_dataset_on_ram()
    # end if
    
    __seq_trial_parameters = []
    for __trial in seq_trial:
        l1 = __trial.suggest_float("l1", search_lower_l1, search_upper_l1)
        l2 = __trial.suggest_float("l2", search_lower_l2, search_upper_l2)
        __seq_trial_parameters.append({"l1": l1, "l2": l2, "id": __trial._trial_id})
    # end for

    __func_dask_task = functools.partial(
        __function_wrapper_optuna,
        dataset_train=dataset_train,
        dataset_test=dataset_dev,  # use dev set for validation
        pytorch_trainer_config=pytorch_trainer_config,
        mmd_estimator=mmd_estimator,
        training_parameter=training_parameter)

    if dask_client is None:
        task_return = [__func_dask_task(__dict_param) for __dict_param in __seq_trial_parameters]
    else:
        logger.debug(f'Executing tasks with Dask...')    
        task_queue = dask_client.map(__func_dask_task, __seq_trial_parameters)
        task_return = dask_client.gather(task_queue)
        logger.debug(f'Finished executing tasks with Dask.')
    # end if

    # DO it on the main thread.
    seq_return_obj = []  # object to return
    dict_trial_id2_trial = {__trial._trial_id: __trial for __trial in seq_trial}
    for t_res in task_return:
        __res_obj = t_res[0]
        __trial_number = t_res[1]
        assert isinstance(__trial_number, int), f'__trial_number = {__trial_number}'
        __trial_obj = dict_trial_id2_trial[__trial_number]

        if __res_obj is None:
            # when the result is None, return invalid objective value.
            invalid_obj_value = __get_invalid_objective_value(__trial_obj, objective_function, __res_obj)
            __o = _DaskFunctionReturn(__trial_obj, invalid_obj_value, invalid_obj_value, [], None, None)
            seq_return_obj.append(__o)
        # end if
        elif torch.isnan(__res_obj.ard_weights_kernel_k).any().item():
            logger.warning(f'NaN detected. trial = {__trial_obj.number}.')
            invalid_obj_value = __get_invalid_objective_value(__trial_obj, objective_function, __res_obj)
            __o = _DaskFunctionReturn(__trial_obj, invalid_obj_value, invalid_obj_value, [], None, None)
            seq_return_obj.append(__o)
        # end if
        elif __res_obj.training_stats.nan_ratio is not None and __res_obj.training_stats.nan_ratio > 0.7:
            logger.warning(f'ratio is always nan. trial = {__trial_obj.number}.')
            invalid_obj_value = __get_invalid_objective_value(__trial_obj, objective_function, __res_obj)
            __o = _DaskFunctionReturn(__trial_obj, invalid_obj_value, invalid_obj_value, [], None, None)
            seq_return_obj.append(__o)
        # end if
        else:
            __variables = detect_variables(variable_weights=__res_obj.ard_weights_kernel_k)
            ratio_variables = len(__variables) / len(__res_obj.ard_weights_kernel_k)
            
            # comment: when upper bound search too large, ARD weights opt ends in inproper shape, and variable detection becomes funny shape.
            # I prevent from being such values.
            if ratio_variable_current_max is not None and ratio_variables > ratio_variable_current_max:
                invalid_obj_value = __get_invalid_objective_value(__trial_obj, objective_function, __res_obj)
                __o = _DaskFunctionReturn(__trial_obj, invalid_obj_value, invalid_obj_value, [], None, None)
                seq_return_obj.append(__o)
            # end if    
            else:
                # running permutation test
                seq_permutation_result_dev = permutation_tests(dataset_dev, 
                                                                variable_selection_approach='hard', 
                                                                interpretable_mmd_result=__res_obj,
                                                                distance_functions=test_distance_functions,
                                                                dask_client=None,
                                                                n_permutation_test=n_permutation_test)
                p_value_max_dev = max([__res_p.p_value for __res_p in seq_permutation_result_dev])
                if dataset_test is not None:
                    seq_permutation_result_test = permutation_tests(dataset_test, 
                                                                    variable_selection_approach='hard', 
                                                                    interpretable_mmd_result=__res_obj,
                                                                    distance_functions=test_distance_functions,
                                                                    dask_client=None,
                                                                    n_permutation_test=n_permutation_test)
                    p_value_max_test = max([__res_p.p_value for __res_p in seq_permutation_result_test])
                else:
                    p_value_max_test = None
                # end if    
                
                optuna_obj_value = __get_optuna_objective_value(
                    trial=__trial_obj,
                    objective_function=objective_function,
                    detection_result=__res_obj,
                    p_value_dev=p_value_max_dev,
                    variables_without_regularization=variables_without_regularization)
                
                assert __res_obj.training_parameter is not None

                # stop counting execution time
                end_cpu_time = time.process_time()
                end_wall_time = timeit.default_timer()
                
                exec_time_wallclock = end_wall_time - start_wall_time
                exec_time_cpu = end_cpu_time - start_cpu_time
                
                epochs = __res_obj.trajectory_record_training[-1].epoch
                test_power_dev = __res_obj.trajectory_record_validation[-1].ratio
                
                res = _DaskFunctionReturn(
                    trial=__trial_obj, 
                    optuna_objective_value=optuna_obj_value,
                    ratio_variable=ratio_variables, 
                    selected_variables=__variables, 
                    epochs=epochs,
                    mmd_train_result=__res_obj,
                    reg_parameter=__res_obj.training_parameter.regularization_parameter,
                    execution_time_wallclock=exec_time_wallclock,
                    execution_time_cpu=exec_time_cpu,
                    test_power_dev=test_power_dev,
                    p_value_dev=p_value_max_dev,
                    p_value_test=p_value_max_test)
                seq_return_obj.append(res)
            # end if
        # end if
    # end for
    return seq_return_obj


# --------------------------------------------------
# utils


def select_best_opt_result(seq_trials: ty.List[_DaskFunctionReturn],
                           search_direction: str) -> ty.Optional[InterpretableMmdTrainResult]:
    """Selecting the best result from the list of optuna trials.
    
    Returns
    -------
    ty.Optional[InterpretableMmdTrainResult]
        The best result. If all trials are None, returns None.
    """	
    assert search_direction in ['minimize', 'maximize'], f'search_direction = {search_direction}'
    # filtering out None results
    seq_trials = [__res for __res in seq_trials if __res.mmd_train_result is not None]
    
    if len(seq_trials) == 0:
        return None
    # end if
    
    if search_direction == 'minimize':
        __seq_sorted = sorted(seq_trials, key=lambda x: x.optuna_objective_value)
        # select the smallest reg. parameter
        ratio_value = __seq_sorted[0].ratio_variable
        assert __seq_sorted[0].mmd_train_result is not None
        return __seq_sorted[0].mmd_train_result
    elif search_direction == 'maximize':
        __seq_sorted = sorted(seq_trials, key=lambda x: x.optuna_objective_value, reverse=True)
        assert __seq_sorted[0].mmd_train_result is not None
        return __seq_sorted[0].mmd_train_result
    else:
        raise NotImplementedError(f'search_direction = {search_direction}')
    
    
def make_execution_statistics(seq_search_result: ty.List[_DaskFunctionReturn]) -> ty.List[ExecutionStatistics]:
    seq_statistics = []
    for _dask_return in seq_search_result:
        assert _dask_return is not None
        if _dask_return.mmd_train_result is not None:                
            assert _dask_return.reg_parameter is not None
            assert _dask_return.epochs is not None
            assert _dask_return.execution_time_wallclock is not None
            assert _dask_return.execution_time_cpu is not None
            
            __exec = ExecutionStatistics(
                regularization_parameters=_dask_return.reg_parameter,
                epochs=_dask_return.epochs,
                execution_time_wall_clock=_dask_return.execution_time_wallclock,
                execution_time_wall_cpu=_dask_return.execution_time_cpu)
            seq_statistics.append(__exec)
        # end if
    # end for
    return seq_statistics


def log_post_process(post_process_handler: PostProcessLoggerHandler, 
                     seq_results_one_batch: ty.List[_DaskFunctionReturn],
                     search_mode: str,
                     cv_detection_experiment_name: str = 'optuna_search'):
    """Private API. Logging post-process results."""
    for __res in seq_results_one_batch:
        if __res.mmd_train_result is None:
            logger.debug(f'{__res.trial.number} is None. Skip logging.')
            continue
        # end if
        
        __run_name = f'{search_mode}-trial-{__res.trial.number}'
        __loggers = post_process_handler.initialize_logger(run_name=__run_name, 
                                                           group_name=cv_detection_experiment_name)
        post_process_handler.log(loggers=__loggers, target_object=__res)
    # end for