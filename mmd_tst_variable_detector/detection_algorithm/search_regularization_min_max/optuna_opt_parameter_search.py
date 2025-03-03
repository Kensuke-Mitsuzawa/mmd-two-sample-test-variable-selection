import typing as ty
import logging
import functools
from collections.abc import Iterable
from pathlib import Path

import optuna

from distributed import Client
from tempfile import mkdtemp

from ...datasets import BaseDataset
from ...mmd_estimator.mmd_estimator import BaseMmdEstimator
from ..pytorch_lightning_trainer import PytorchLightningDefaultArguments
from ..interpretable_mmd_detector import InterpretableMmdTrainParameters
from ..commons import RegularizationParameter
from ...utils import (
    PostProcessLoggerHandler
)
from .optuna_module.optuna_search_module_core import (
    _DaskFunctionReturn,
    func_dask_weapper_function_optuna,
    select_best_opt_result,
    make_execution_statistics,
    log_post_process)
from .optuna_module.commons import SelectionResult
from ...exceptions import ParameterSearchException
from ...logger_unit import handler


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)



def __run_parameter_space_search(study: optuna.Study,
                                 dataset_train: BaseDataset,
                                 dataset_dev: BaseDataset,
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
                                 dataset_test: ty.Optional[BaseDataset] = None,
                                 test_distance_functions: ty.Tuple[str, ...] = ('sliced_wasserstein',),
                                 variables_without_regularization: ty.Optional[ty.List[int]] = None,
                                 n_permutation_test: int = 500
                                 ) -> ty.List[_DaskFunctionReturn]:
    """A function to search the regularization parameter. 
    This function defines the procedure of the search.
    
    Args
    ----
    study: optuna.Study
        Optuna study object.
    dataset_train: BaseDataset
        Training dataset.
    dataset_test: BaseDataset
        Testing dataset.        
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
    test_distance_functions: ty.Optional[ty.List[str]]
    variables_without_regularization: ty.Optional[ty.List[int]]
        Used only for the objective value of Optuna.
        When detected_variables > N(variables_without_regularization), then the objective value is set to a big value.
    """
    logger.debug(f'concurrent_limit = {concurrent_limit}')
    
    # stack to save
    stack_opt_result = []
    
    # parameter search space
    search_lower_l1 = reg_param_lower.lambda_1
    search_lower_l2 = reg_param_lower.lambda_2
    search_upper_l1 = reg_param_upper.lambda_1
    search_upper_l2 = reg_param_upper.lambda_2

    current_trials = 0
    while current_trials < n_trials:
        logger.debug(f'current_trials = {current_trials}')
        
        __seq_trial_stack = [study.ask() for __i in range(concurrent_limit)]
        task_return = func_dask_weapper_function_optuna(
            seq_trial=__seq_trial_stack,
            objective_function='testpower_pvalue',
            path_work_dir=path_opt_result,
            dataset_train=dataset_train,
            dataset_dev=dataset_dev,
            dataset_test=dataset_test,
            pytorch_trainer_config=pytorch_trainer_config,
            search_lower_l1=search_lower_l1,
            search_upper_l1=search_upper_l1,
            search_lower_l2=search_lower_l2,
            search_upper_l2=search_upper_l2,
            mmd_estimator=mmd_estimator,
            training_parameter=training_parameter,
            test_distance_functions=test_distance_functions,
            variables_without_regularization=variables_without_regularization,
            n_permutation_test=n_permutation_test)
        assert task_return is not None
        assert isinstance(task_return, Iterable)
        
        # criteria. If MMD-estimates are all super small or minues, reset the lambda upper bound.
        _seq_mmd_estimate = []
        
        # reporting the Optuna objective value to Optuna.
        for __t_return in task_return:
            __trial = __t_return.trial
            __eval_score = __t_return.optuna_objective_value        
            study.tell(__trial, __eval_score)
            
            # adding the result to the stack
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
            if all([_mmd < 0.0001 for _mmd in _seq_mmd_estimate]):
                search_upper_l1 = search_upper_l1 * 0.1
                search_upper_l2 = search_upper_l2 * 0.1
                search_lower_l1 = search_lower_l1 * 0.1
                search_lower_l2 = search_lower_l2 * 0.1
                logger.debug(f'All MMD estimates are super small. Resetting the upper bound of lambda: {search_upper_l1}, {search_upper_l2}')
            # end if
        # end for
        current_trials += concurrent_limit
    # end while

    return stack_opt_result


def main(dataset_train: BaseDataset,
         dataset_dev: BaseDataset,
         mmd_estimator: BaseMmdEstimator,
         base_training_parameter: InterpretableMmdTrainParameters,
         pytorch_trainer_config: PytorchLightningDefaultArguments,
         dataset_test: ty.Optional[BaseDataset] = None,
         path_storage_backend_db: ty.Optional[Path] = None,
         path_work_dir: ty.Optional[Path] = None,
         dask_client: ty.Optional[Client] = None,
         post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
         n_trials: int = 20,
         concurrent_limit: int = 1,
         regularization_param_search_upper: RegularizationParameter = RegularizationParameter(2.0, 0.0),
         regularization_param_search_lower: RegularizationParameter = RegularizationParameter(0.0001, 0.0),
         test_distance_functions: ty.Tuple[str, ...] = ('sliced_wasserstein',),
         variables_without_regularization: ty.Optional[ty.List[int]] = None,
         n_permutation_test: int = 500,
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
    dataset_test: BaseDataset
        Test dataset.        
    mmd_estimator: BaseMmdEstimator
    base_training_parameter: InterpretableMmdTrainParameters
    pytorch_trainer_config: PytorchLightningDefaultArguments
    path_storage_backend_db: ty.Optional[Path]
        Path to the storage backend database.
    path_work_dir: ty.Optional[Path]
        Path to the working directory.
    dask_client: ty.Optional[Client]
        Dask client.
    n_trials: int
        Number of trials.
    concurrent_limit: int
        Number of concurrent trials.
    regularization_param_search_upper: RegularizationParameter
        Upper bound of regularization parameter.
    regularization_param_search_lower: RegularizationParameter
        Lower bound of regularization parameter.
    test_distance_functions: ty.Tuple[str, ...]
        Distance functions to test.
    variables_without_regularization: ty.Optional[ty.List[int]]
        Variables used a criteria for the objective value of Optuna.
        When detected_variables > N(variables_without_regularization), then the objective value is set to infinity.
    """
    
    if path_work_dir is None:
        path_work_dir =  Path(mkdtemp()) / 'tst_based_regression_tuner'
        path_work_dir.mkdir(parents=True, exist_ok=True)
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
    study_lower = optuna.create_study(storage=__path_storage_backend_db, direction="minimize")
        
    __path_opt_result = path_work_dir / 'optuna-lambda-search'
    
    seq_optuna_search_result = __run_parameter_space_search(
        study=study_lower,
        dataset_train=dataset_train,
        dataset_dev=dataset_dev,
        dataset_test=dataset_test,
        pytorch_trainer_config=pytorch_trainer_config,
        reg_param_lower=regularization_param_search_lower,
        reg_param_upper=regularization_param_search_upper,
        mmd_estimator=mmd_estimator,
        training_parameter=base_training_parameter,
        n_trials=n_trials,
        concurrent_limit=concurrent_limit,
        search_mode='optuna-lambda-search',
        path_opt_result=__path_opt_result,
        dask_client=dask_client,
        post_process_handler=post_process_handler,
        test_distance_functions=test_distance_functions,
        variables_without_regularization=variables_without_regularization,
        n_permutation_test=n_permutation_test)

    # -------------------------------------------------------------------    
    # post processing
        
    # selecting the best result
    __seq_t_optuna_obj_value_pair = [
        (__func_out.optuna_objective_value, __func_out) for __func_out in seq_optuna_search_result 
        if __func_out.mmd_train_result is not None]
    if len(__seq_t_optuna_obj_value_pair) == 0:
        raise ParameterSearchException(f'All search results are None, failed to run executions. Multiple reasons.')
    # end if
    
    selected_func_out = sorted(__seq_t_optuna_obj_value_pair, key=lambda x: x[0])[0][1]
    # selected_variables = selected_func_out.selected_variables
    
    selected_variables = [
        (__func_out.reg_parameter, __func_out.selected_variables) for __func_out in seq_optuna_search_result
        if __func_out.reg_parameter is not None
    ]
    
    if len(selected_variables) == 0:
        raise ParameterSearchException(f'All search results are None, failed to run executions. Multiple reasons.')
    # end if
    
    dict_regularization2model_parameter = {
        _dask_return.mmd_train_result.training_parameter.regularization_parameter: _dask_return.mmd_train_result.mmd_estimator
        for _dask_return in seq_optuna_search_result
        if _dask_return.mmd_train_result is not None and _dask_return.mmd_train_result.training_parameter is not None
    }
    dict_regularization2optuna_return = {
        _dask_return.mmd_train_result.training_parameter.regularization_parameter: _dask_return
        for _dask_return in seq_optuna_search_result
        if _dask_return.mmd_train_result is not None and _dask_return.mmd_train_result.training_parameter is not None
    }    
    seq_reg_parameters = [reg for reg in dict_regularization2model_parameter.keys()]
    seq_l1 = [reg.lambda_1 for reg in seq_reg_parameters]
    seq_l2 = [reg.lambda_2 for reg in seq_reg_parameters]
    
    lower_bound = RegularizationParameter(min(seq_l1), min(seq_l2))
    upper_bound = RegularizationParameter(max(seq_l1), max(seq_l2))
    
    execution_statistics = make_execution_statistics(seq_optuna_search_result)
    
    assert dict_regularization2optuna_return is not None
    result = SelectionResult(
        regularization_parameters=seq_reg_parameters,
        selected_variables=selected_variables,
        regularization_upper_searched=upper_bound,
        regularization_lower_searched=lower_bound,
        execution_statistics=execution_statistics,
        dict_regularization2model_parameter=dict_regularization2model_parameter,
        dict_regularization2optuna_return=dict_regularization2optuna_return,
    )
    return result
