from collections.abc import Iterable
import logging
import functools
import typing as ty
import copy
import traceback
from pathlib import Path
from tempfile import mkdtemp
from dataclasses import asdict

import optuna

from distributed import Client

import torch
import numpy as np
import pytorch_lightning as pl

from ...utils.post_process_logger import PostProcessLoggerHandler
from ...utils import detect_variables
from ...mmd_estimator.mmd_estimator import BaseMmdEstimator
from ...datasets import BaseDataset
from ...exceptions import ParameterSearchException
from ..pytorch_lightning_trainer import PytorchLightningDefaultArguments
from ..interpretable_mmd_detector import InterpretableMmdDetector
from ..commons import (
    RegularizationParameter, 
    InterpretableMmdTrainParameters, 
    InterpretableMmdTrainResult)
from .optuna_module.commons import SelectionResult, ExecutionStatistics
from .optuna_module.optuna_search_module_core import (
    _DaskFunctionReturn,
    make_execution_statistics,
    func_dask_weapper_function_optuna,
    select_best_opt_result,
    log_post_process)

from ...exceptions import OptimizationException, ParameterSearchException


logger = logging.getLogger(f'{__package__}.{__name__}')



def __get_maximum_regularization_parameter_from_optuna(seq_trials: ty.List[_DaskFunctionReturn]) -> RegularizationParameter:
    """Get the maximum regularization parameter from the Optuna search results.
    This function is designes for the results of upper-bound search."""
    trial_sequences = [trial.reg_parameter for trial in seq_trials 
                       if len(trial.selected_variables) > 0]
    lambda_one_max = max([reg.lambda_1 for reg in trial_sequences])
    lambda_two_max = max([reg.lambda_2 for reg in trial_sequences])
    
    return RegularizationParameter(lambda_1=lambda_one_max, lambda_2=lambda_two_max)


def __run_parameter_space_search(study: optuna.Study,
                                 dataset_all: BaseDataset,
                                 pytorch_trainer_config: PytorchLightningDefaultArguments,
                                 reg_param_lower: RegularizationParameter,
                                 reg_param_upper: RegularizationParameter,
                                 mmd_estimator: BaseMmdEstimator,
                                 training_parameter: InterpretableMmdTrainParameters,                                 
                                 n_trials: int,
                                 concurrent_limit: int,
                                 path_opt_result: Path,
                                 search_mode: str,
                                 dask_client: ty.Optional[Client] = None,
                                 post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
                                 ratio_variable_current_max: ty.Optional[float] = None
                                 ) -> ty.List[_DaskFunctionReturn]:
    """
    Args
    ----
    study: optuna.Study
        Optuna study object.
    dataset_all: BaseDataset
        Training dataset.
    pytorch_trainer_config: PytorchLightningDefaultArguments
        Pytorch lightning trainer configuration.
    reg_param_lower: ty.Tuple[float, float]
        Lower bound of the regularization parameter.
    reg_param_upper: ty.Tuple[float, float]
        Upper bound of the regularization parameter.
    mmd_estimator: BaseMmdEstimator
        MMD estimator.
    training_parameter: InterpretableMmdTrainParameters
        Training parameter.
    n_trials: int
        Number of trials.
    concurrent_limit: int
        Number of concurrent trials.
    path_opt_result: Path
        Path to the result directory.
    dask_client: ty.Optional[Client]
        Dask client.
    post_process_handler: ty.Optional[PostProcessLoggerHandler]
        Post process handler.
    ratio_variable_current_max: ty.Optional[float]
        Used for only upper-bound search.
        If the MMD-opt found variables more than this ratio, value, the optuna function returns -1.
    """
    logger.debug(f'concurrent_limit = {concurrent_limit}')

    # Manual Cross-Validation
    # seq_sample_id = range(dataset_all.__len__())
    # kf_splitter = KFold(n_splits=n_cv)
    # pair_xy_train_test_sample_id = list(kf_splitter.split(seq_sample_id))
    
    # stack to save
    stack_opt_result = []
    
    if search_mode == 'lower-search':
        objective_function = 'ratio_maximum'
    elif search_mode == 'upper-search':
        objective_function = 'ratio_minimum'
    else:
        raise OptimizationException(f'Unknown search mode: {search_mode}')
    # end if
    
    current_upper_l1 = reg_param_upper.lambda_1
    current_upper_l2 = reg_param_upper.lambda_2
    current_lower_l1 = reg_param_lower.lambda_1
    current_lower_l2 = reg_param_lower.lambda_2
    
    current_trials = 0
    while current_trials < n_trials:
        logger.debug(f'current_trials = {current_trials}')

        __seq_trial_stack = [study.ask() for __i in range(concurrent_limit)]
        task_return = func_dask_weapper_function_optuna(
            seq_trial=__seq_trial_stack,
            objective_function=objective_function,
            path_work_dir=path_opt_result,
            dataset_train=dataset_all,
            dataset_dev=dataset_all,
            pytorch_trainer_config=pytorch_trainer_config,
            search_lower_l1=current_lower_l1,
            search_upper_l1=current_upper_l1,
            search_lower_l2=current_lower_l2,
            search_upper_l2=current_upper_l2,
            mmd_estimator=mmd_estimator,
            ratio_variable_current_max=ratio_variable_current_max,
            training_parameter=training_parameter,
            dask_client=dask_client
        )
        
        assert task_return is not None
        assert isinstance(task_return, Iterable)
        
        # criteria. If MMD-estimates are all super small or minues, reset the lambda upper bound.
        _seq_mmd_estimate = []
                
        for __t_return in task_return:
            __trial = __t_return.trial
            __eval_score = __t_return.ratio_variable
            
            study.tell(__trial, __eval_score)
            
            stack_opt_result.append(__t_return)
            
            # collecting the MMD estimate values.
            if __t_return.mmd_train_result is None:
                _seq_mmd_estimate.append(-1.0)
            else:
                _seq_mmd_estimate.append(__t_return.mmd_train_result.trajectory_record_training[-1].mmd)
            # end if            

            # posting results to logger object.
            if post_process_handler is not None:
                log_post_process(
                    post_process_handler=post_process_handler,
                    search_mode=search_mode,
                    seq_results_one_batch=[__t_return])
            # end if
            
            # If MMD-estimates are all super small or minues, reset the lambda upper bound
            # the lambda upper bound is udapted with current * 0.1.
            # if all([_mmd < 0.0001 for _mmd in _seq_mmd_estimate]):
            #     current_upper_l1 = current_upper_l1 * 0.1
            #     current_upper_l2 = current_upper_l2 * 0.1
            #     current_lower_l1 = current_lower_l1 * 0.1
            #     current_lower_l2 = current_lower_l2 * 0.1
            #     logger.debug(f'All MMD estimates are super small. Resetting the upper bound of lambda: {search_upper_l1}, {search_upper_l2}')
            # end if            
        # end for
        current_trials += concurrent_limit
    # end while

    return stack_opt_result


def __generate_regularization_parameters(lower_bound: RegularizationParameter,
                                         upper_bound: RegularizationParameter,
                                         n_regularization_parameter: int
                                         ) -> ty.List[RegularizationParameter]:
    # TODO separating to a function.
    # computing the regularization_parameters candidates step=(max - min) / N-candidate
    # result_best_max.config['l1']
    
    l1_upper = upper_bound.lambda_1
    l2_upper = upper_bound.lambda_2

    l1_lower = lower_bound.lambda_1
    l2_lower = lower_bound.lambda_2

    step_l1 = (l1_upper - l1_lower) / n_regularization_parameter
    step_l2 = (l2_upper - l2_lower) / n_regularization_parameter

    if (l1_upper - l1_lower) > 0.0:
        l1_parameters = np.arange(
            l1_lower,
            l1_upper,
            step=step_l1)
    else:
        l1_parameters = None
    # end if

    if (l2_upper - l2_lower) > 0.0:
        l2_parameters = np.arange(
            l2_lower,
            l2_upper,
            step=step_l2)
    else:
        l2_parameters = None
    # end if

    if l1_parameters is None:
        l1_parameters = np.zeros(len(l2_parameters))
    if l2_parameters is None:
        l2_parameters = np.zeros(len(l1_parameters))
    # end if

    parameters_longer = max([len(l1_parameters), len(l2_parameters)])
    arrays_parameter = np.zeros((2, parameters_longer))

    arrays_parameter[0, :] = l1_parameters
    arrays_parameter[1, :] = l2_parameters

    regularization_parameters = [
        RegularizationParameter(arrays_parameter[0, __i_column], 
                                arrays_parameter[1, __i_column])
        for __i_column in range(arrays_parameter.shape[1])]
    # comment: #189
    # # add min
    # regularization_parameters.append(RegularizationParameter(l1_lower, l2_lower))
    # add max
    regularization_parameters.append(RegularizationParameter(l1_upper, l2_upper))

    return regularization_parameters


def __execute_mmd_opt_without_regularization(dataset_all: BaseDataset,
                                             mmd_estimator: BaseMmdEstimator,
                                             base_training_parameter: InterpretableMmdTrainParameters,
                                             pytorch_trainer_config: PytorchLightningDefaultArguments,
                                             regularization_param_search_lower: RegularizationParameter
                                             ) -> ty.Tuple[ty.List[_DaskFunctionReturn], InterpretableMmdTrainResult]:
    """Execute the MMD-opt without regularization parameter.
    This function is used for the lower-bound search.
    """
    try:
        if dataset_all.is_dataset_on_ram():
            __dataset_all = dataset_all.generate_dataset_on_ram()
        else:
            __dataset_all = dataset_all
        # end if
        new_training_param = copy.deepcopy(base_training_parameter)
        new_training_param.regularization_parameter = RegularizationParameter(0.0, 0.0)
        variable_trainer = InterpretableMmdDetector(
            mmd_estimator=mmd_estimator,
            training_parameter=new_training_param,
            dataset_train=__dataset_all,
            dataset_validation=__dataset_all)
        trainer_pl = pl.Trainer(**asdict(pytorch_trainer_config))
        trainer_pl.fit(variable_trainer)
    except OptimizationException as e:
        raise OptimizationException(f'OptimizationException: {e}.'
                                    'Either inproper regularization parameter or the data-pair (X, Y) are same')
    except Exception as e:
        logger.warning(f'Failed to run the non-regularization optimization. {e}')
        raise Exception(f'Exception: {e}. Traceback -> {traceback.format_exc()}')
    else:
        __trained_result = variable_trainer.get_trained_variables()
        __variables = detect_variables(variable_weights=__trained_result.ard_weights_kernel_k)
        ratio_variables = len(__variables) / len(__trained_result.ard_weights_kernel_k)
        seq_search_result_lower_bound = [_DaskFunctionReturn(
            trial=None,
            optuna_objective_value=np.nan,
            ratio_variable=ratio_variables,
            selected_variables=__variables,
            mmd_train_result=__trained_result,
            reg_parameter=regularization_param_search_lower,
            epochs=__trained_result.trajectory_record_training[-1].epoch,
            test_power_dev=__trained_result.trajectory_record_training[-1].ratio)]
        lower_bound_result = __trained_result
        assert lower_bound_result.training_parameter is not None
        lower_bound_result.training_parameter.regularization_parameter = regularization_param_search_lower
    # end if
    return seq_search_result_lower_bound, lower_bound_result


def run_parameter_space_search(dataset_train: BaseDataset,
                               mmd_estimator: BaseMmdEstimator,
                               base_training_parameter: InterpretableMmdTrainParameters,
                               pytorch_trainer_config: PytorchLightningDefaultArguments,
                               dataset_test: ty.Optional[BaseDataset] = None,
                               path_storage_backend_db: ty.Optional[Path] = None,
                               path_work_dir: ty.Optional[Path] = None,
                               dask_client: ty.Optional[Client] = None,
                               post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
                               n_regularization_parameter: int = 4,
                               n_trials: int = 20,
                               concurrent_limit: int = 1,
                               regularization_param_search_upper: RegularizationParameter = RegularizationParameter(1.5, 0.0),
                               regularization_param_search_lower: RegularizationParameter = RegularizationParameter(0.0001, 0.0),
                               initial_regularization_search_search_lower: RegularizationParameter = RegularizationParameter(0.01, 0.0),
                               initial_regularization_search_search_upper: ty.Union[str, RegularizationParameter] = 'auto',
                               is_use_search_search_lower_exception: bool = True,
                               is_use_non_regularization_opt_as_lower_bound: bool = True
                               ) -> SelectionResult:
    """Search for the regularization parameter. It searches the lower and upper bound of regularization parameters.
    The parameter search function utilises Optuna. 
    A distributed platflorm, dask, promotes efficiently the search.
    
    Since Optuna can utilize search histories, you can import your own search history. `path_storage_backend_db`.
    The search history will find more accurate regularization parameters in a fewer number of trials.
    
    Parameters
    ------------
    dataset_train: BaseDataset
        Training dataset.
    mmd_estimator: BaseMmdEstimator
    base_training_parameter: InterpretableMmdTrainParameters
    pytorch_trainer_config: PytorchLightningDefaultArguments
    dataset_test: ty.Optional[BaseDataset]
        Test dataset.
    path_storage_backend_db: ty.Optional[Path]
        Path to the storage backend database.
    path_work_dir: ty.Optional[Path]
        Path to the working directory.
    dask_client: ty.Optional[Client]
        Dask client.
    n_regularization_parameter: int
        Number of regularization parameters to search.
    n_trials: int
        Number of trials.
    concurrent_limit: int
        Number of concurrent trials.
    regularization_param_search_upper: RegularizationParameter
        Upper bound of regularization parameter.
    regularization_param_search_lower: RegularizationParameter
        Lower bound of regularization parameter.
    initial_regularization_search_search_lower: RegularizationParameter
        The maximum regularization parameter value of Optuna search for the lower bound.
    initial_regularization_search_search_upper: ty.Union[str, RegularizationParameter]
        Default is 'auto'. The 'auto' mode employs the maximum regularization parameter value of Optuna search from results of the lower bound search.
        If you wanna set the value manually, you can set a RegularizationParameter object.
    is_use_search_search_lower_exception: bool
        This option is used for the upper bound search.
        If True, the search will be stopped when all trials are None.
        If False, the search will be continued until the number of trials reaches n_trials.
    is_use_non_regularization_opt_as_lower_bound: bool
        If True, the module uses the non-regularization optimization result as the lower bound.
    """
    
    if path_work_dir is None:
        path_work_dir =  Path(mkdtemp()) / 'optuna_upper_lower_search'
        path_work_dir.mkdir(parents=True, exist_ok=True)
    # end if
    
    # merging two dataset into one.
    if dataset_test is None:
        dataset_all = dataset_train
    else:
        logger.debug(f'Merging two datasets: {dataset_train} and {dataset_test}')
        dataset_all = dataset_train.merge_new_dataset(dataset_test)
    # end if
    
    assert path_work_dir.exists(), f'path_work_dir does not exist: {path_work_dir}'
    # -------------------------------------------------------------------
    
    if path_storage_backend_db is None:
        __path_storage_backend_db = f'sqlite:///{path_work_dir / "optuna.sqlite3"}'
    else:
        __path_storage_backend_db = f'sqlite:///{path_storage_backend_db}'
    # end if
    logger.debug(f'Optuna backend DB is at {__path_storage_backend_db}')
    
    # -------------------------------------------------------------------
    # search for lower-bound
    
    if is_use_non_regularization_opt_as_lower_bound:
        # I use the non-regularization optimization result as the lower bound.
        seq_search_result_lower_bound, lower_bound_result = __execute_mmd_opt_without_regularization(
            dataset_all=dataset_all,
            mmd_estimator=mmd_estimator,
            base_training_parameter=base_training_parameter,
            pytorch_trainer_config=pytorch_trainer_config,
            regularization_param_search_lower=regularization_param_search_lower)
    else:
        study_lower = optuna.create_study(storage=__path_storage_backend_db, direction="maximize")
            
        __path_opt_result = path_work_dir / 'lower'
        
        seq_search_result_lower_bound = __run_parameter_space_search(
            study=study_lower,
            dataset_all=dataset_all,
            pytorch_trainer_config=pytorch_trainer_config,
            reg_param_lower=regularization_param_search_lower,
            reg_param_upper=initial_regularization_search_search_lower,
            mmd_estimator=mmd_estimator,
            training_parameter=base_training_parameter,
            n_trials=n_trials,
            concurrent_limit=concurrent_limit,
            search_mode='lower-search',
            path_opt_result=__path_opt_result,
            dask_client=dask_client,
            post_process_handler=post_process_handler)

        lower_bound_result = select_best_opt_result(seq_search_result_lower_bound, search_direction='maximize')
        # comment: lower bound search is not supposed to be None.
        assert lower_bound_result is not None
        assert lower_bound_result.training_parameter is not None
    # end if
    # -------------------------------------------------------------------
    # search for upper-bound
    study_upper = optuna.create_study(storage=__path_storage_backend_db, direction="minimize")
    
    if initial_regularization_search_search_upper == 'auto':
        try:
            __min_reg_search_upper = __get_maximum_regularization_parameter_from_optuna(seq_search_result_lower_bound)
        except ValueError as e:
            raise ParameterSearchException('Failed to get variables. This is often because of P=Q.')
    else:
        assert isinstance(initial_regularization_search_search_upper, RegularizationParameter)
        __min_reg_search_upper = initial_regularization_search_search_upper
    # end if
    
    # comment:
    ratio_variable_max_current = max([
        __func_result.ratio_variable for __func_result in seq_search_result_lower_bound 
        if __func_result.mmd_train_result is not None])    
    
    __path_opt_result = path_work_dir / 'upper'
    
    seq_search_result_upper_bound = __run_parameter_space_search(
        study=study_upper,
        dataset_all=dataset_all,
        pytorch_trainer_config=pytorch_trainer_config,
        reg_param_lower=__min_reg_search_upper,
        reg_param_upper=regularization_param_search_upper,
        mmd_estimator=mmd_estimator,
        training_parameter=base_training_parameter,
        n_trials=n_trials,
        concurrent_limit=concurrent_limit,
        search_mode='upper-search',
        path_opt_result=__path_opt_result,
        dask_client=dask_client,
        ratio_variable_current_max=ratio_variable_max_current,
        post_process_handler=post_process_handler)

    # comment: 
    # when all trials are None, use the minimum lambda value as the upper bound.
    upper_bound_result = select_best_opt_result(seq_search_result_upper_bound, search_direction='minimize')
    if upper_bound_result is None:
        if is_use_search_search_lower_exception and upper_bound_result is None:
            logger.warning(f'All trials are None in searching upper-bound. Use the lower-bound as the upper-bound.')
            upper_bound = initial_regularization_search_search_lower
        else:
            raise OptimizationException(
                f'All trials are None in searching upper-search.' 
                'Hint: increase n_trials or check your MMD-opt configuration.')
    else:
        assert upper_bound_result is not None
        assert upper_bound_result.training_parameter is not None        
        upper_bound = upper_bound_result.training_parameter.regularization_parameter
        logger.debug(f'upper_bound = {upper_bound}')
    # end if
    
    # -------------------------------------------------------------------    
    
    lower_bound = lower_bound_result.training_parameter.regularization_parameter    
    logger.debug(f'lower_bound = {lower_bound}')

    # -------------------------------------------------------------------    
    # post processing
    if upper_bound_result is None:
        seq_search_result = seq_search_result_lower_bound
    else:
        seq_search_result = seq_search_result_lower_bound + seq_search_result_upper_bound
    # end if

    seq_reg_parameters = __generate_regularization_parameters(
        lower_bound=lower_bound, 
        upper_bound=upper_bound,
        n_regularization_parameter=n_regularization_parameter)
    
    # generation of selected_variables
    selected_variables = [
        (_dask_return.mmd_train_result.training_parameter.regularization_parameter, _dask_return.selected_variables)
        for _dask_return in seq_search_result
        if _dask_return.mmd_train_result is not None and _dask_return.mmd_train_result.training_parameter is not None
    ]
    
    if len(selected_variables) == 0:
        raise ParameterSearchException(f'All search results are None, failed to run executions. Multiple reasons.')
    # end if
    
    dict_regularization2model_parameter = {
        _dask_return.mmd_train_result.training_parameter.regularization_parameter: _dask_return.mmd_train_result.mmd_estimator
        for _dask_return in seq_search_result
        if _dask_return.mmd_train_result is not None and _dask_return.mmd_train_result.training_parameter is not None
    }
    
    execution_statistics = make_execution_statistics(seq_search_result_upper_bound)
        
    result = SelectionResult(
        regularization_parameters=seq_reg_parameters,
        selected_variables=selected_variables,
        dict_regularization2model_parameter=dict_regularization2model_parameter,
        regularization_upper_searched=upper_bound,
        regularization_lower_searched=lower_bound,
        execution_statistics=execution_statistics
    )
    return result
