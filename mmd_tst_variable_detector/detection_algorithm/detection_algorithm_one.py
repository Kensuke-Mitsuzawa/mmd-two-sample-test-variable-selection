import more_itertools
import typing as ty
import shutil
import time
import logging
from copy import deepcopy
from pathlib import Path
from distributed import Client
from tempfile import mkdtemp
from dataclasses import dataclass

import ot
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger

from .. import logger_unit
# detector class sample selection based detector
from ..utils.post_process_logger import log_postprocess_manually
from .utils import permutation_tests
from ..mmd_estimator import BaseMmdEstimator
from ..datasets import BaseDataset
from ..datasets.file_onetime_load_backend_static_dataset import FileBackendOneTimeLoadStaticDataset
from ..utils import (
    PostProcessLoggerHandler, 
    detect_variables, 
    PermutationTest)

from .pytorch_lightning_trainer import PytorchLightningDefaultArguments
from .interpretable_mmd_detector import (
    InterpretableMmdTrainResult, 
    InterpretableMmdTrainParameters)
from .commons import RegularizationParameter
from .search_regularization_min_max import RegularizationSearchParameters, optuna_opt_parameter_search
from .interpretable_mmd_detector import InterpretableMmdDetector
from .search_regularization_min_max import (
    optuna_search, 
    heuristic_search,
    SelectionResult)


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(logger_unit.handler)


"""A module for ARD weights optimization. 
This module corresponds to the approach 1 in the paper.
This module employs Dask for the parallel computation.
"""


from pytorch_lightning.loggers import Logger
from .search_regularization_min_max import SelectionResult


@dataclass
class AlgorithmOneIndividualResult:
    regularization_parameter: RegularizationParameter
    selected_variables: ty.List[int]
    trained_ard_weights: ty.Optional[ty.Union[np.ndarray, torch.Tensor]]
    p_value_dev: ty.Optional[float]
    test_power_dev: ty.Optional[float]
    p_value_test: ty.Optional[float] = None
    pl_loggers: ty.Optional[ty.List[Logger]] = None  # saving logger object. So, a user can do logging operation later.
    interpretable_mmd_train_result: ty.Optional[InterpretableMmdTrainResult] = None
    

@dataclass
class AlgorithmOneResult:
    selected_model: ty.Optional[AlgorithmOneIndividualResult]
    trained_models: ty.List[AlgorithmOneIndividualResult]
    param_search_result: ty.Optional[SelectionResult] = None


# ------------------------------------------------------------
# possible types for arguments

PossibleTypeRegularizationParameter = ty.Union[str, SelectionResult, ty.List[RegularizationParameter]]



# ----------------------------------------------
# aux clas definitions

class _AlgorithmOneRangeFunctionRequestPayload(ty.NamedTuple):
    # distributed computation parameter class. Private.
    regularization_parameter: RegularizationParameter
    mmd_estimator: BaseMmdEstimator
    parameter_variable_trainer: InterpretableMmdTrainParameters
    dataset_training: BaseDataset
    dataset_dev: BaseDataset
    pl_trainer_config: PytorchLightningDefaultArguments
    dataset_test: ty.Optional[BaseDataset] = None
    permutation_test_runner: ty.Optional[PermutationTest] = None
    variable_detection_method: str = "hist_based"
    path_work_dir: ty.Optional[Path] = None
    pl_loggers: ty.Optional[ty.List[Logger]] = None
    test_distance_functions: ty.Tuple[str, ...] = ('sliced_wasserstein',)
    n_permutation_test: int = 500


class _AlgorithmOneRangeFunctionReturn(ty.NamedTuple):
    # return of distributed computation. Private.
    request: _AlgorithmOneRangeFunctionRequestPayload
    trained_result: ty.Optional[InterpretableMmdTrainResult]
    indices_detected: ty.List[int]
    regularization_parameter: RegularizationParameter
    trainer: ty.Optional[pl.Trainer] = None
    test_power_dev: ty.Optional[float] = None
    p_value_dev: ty.Optional[float] = None
    p_value_test: ty.Optional[float] = None
    is_success: ty.Optional[bool] = False
    execution_time_wallclock: ty.Optional[float] = None
    execution_time_cpu: ty.Optional[float] = None
    epochs: ty.Optional[int] = None
    
    def get_key_id(self) -> str:
        __l1 = self.request.regularization_parameter[0]
        __l2 = self.request.regularization_parameter[1]
        return f'{__l1}_{__l2}'


# ----------------------------------------------
# algorithm definitions

# TODO in the future I want to move this function.
# I want to unify and make it common function among various approaches.
# def func_distance_sliced_wasserstein_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     if isinstance(x, torch.Tensor):
#         x = x.cpu().detach().numpy()
#     if isinstance(y, torch.Tensor):
#         y = y.cpu().detach().numpy()
    
#     v = ot.sliced_wasserstein_distance(x, y)
#     return torch.tensor(v) 
# # end def



def __run_optimization_estimator(requests: _AlgorithmOneRangeFunctionRequestPayload) -> _AlgorithmOneRangeFunctionReturn:
    """Executing optimization of an MMD estimator.
    Based on the optimization resut, this function selects variables.
    Then, this function executes permutation test for the selected variables.
    """
    start_exec = time.time()
    
    # n_permutation_test = requests.n_permutation_test  # TODO delete this line.
    
    pl_trainer = pl.Trainer(**requests.pl_trainer_config.as_dict())
    
    if requests.dataset_training.is_dataset_on_ram():
        dataset_training = requests.dataset_training.generate_dataset_on_ram()
    else:
        dataset_training = requests.dataset_training
    if requests.dataset_dev.is_dataset_on_ram():
        dataset_dev = requests.dataset_dev.generate_dataset_on_ram()
    else:
        dataset_dev = requests.dataset_dev
    # end if
    
    mmd_variable_trainer = InterpretableMmdDetector(
        mmd_estimator=requests.mmd_estimator,
        training_parameter=requests.parameter_variable_trainer,
        dataset_train=dataset_training,
        dataset_validation=dataset_dev)
    
    pl_trainer.fit(mmd_variable_trainer)
    trained_result = mmd_variable_trainer.get_trained_variables()
    seq_selected_variable = detect_variables(variable_weights=trained_result.ard_weights_kernel_k, 
                                             variable_detection_approach=requests.variable_detection_method)


    if isinstance(trained_result.training_stats.nan_ratio, float) and trained_result.training_stats.nan_ratio > 0.9:
        # if block when the optimization failed.
        return _AlgorithmOneRangeFunctionReturn(
            request=requests,
            trained_result=trained_result,
            indices_detected=[],
            regularization_parameter=RegularizationParameter(*requests.regularization_parameter),
            trainer=pl_trainer,
            is_success=False,
            execution_time_cpu=0.0,
            execution_time_wallclock=0.0,
            epochs=0,)
    # end if

    # TODO
    """Permutation Testについて、次のような設計
    
    Permutation Test objがgiven -> そのまま実行
    Permutation Test objがNone -> 選択された変数のみをで、MMD estimatorのoptimizationを再実行。
    ２つのoptionを用意する。1. full sctratchでoptimization, 2. 用意されたARD weightsを初期値としてoptimization
    """
    # selected variables based on the optimized result.
    # dataset_dev_select = dataset_dev.get_selected_variables_dataset(seq_selected_variable)
    
    # Permutation Test for the dev. data
    # permutation_runner = requests.permutation_test_runner
    # if permutation_runner is None:
    #     permutation_runner = PermutationTest(
    #         func_distance=func_distance_sliced_wasserstein_distance,  # type: ignore
    #         batch_size=mmd_variable_trainer.training_parameter.batch_size,
    #         n_permutation_test=n_permutation_test)
    # # enf if
    # p_dev, __ = permutation_runner.run_test(dataset_dev_select)
    seq_dev_tst_result_container = permutation_tests.permutation_tests(
        dataset_test=dataset_dev,
        variable_selection_approach='hard',
        interpretable_mmd_result=trained_result,
        n_permutation_test=requests.n_permutation_test,
        distance_functions=requests.test_distance_functions,
    )
    p_dev = np.max([__tst_obj.p_value for __tst_obj in seq_dev_tst_result_container])


    # permutation test for test data    
    if requests.dataset_test is not None:
        if requests.dataset_test.is_dataset_on_ram():
            dataset_test = requests.dataset_test.generate_dataset_on_ram()
        else:
            dataset_test = requests.dataset_test
        # end if
        
        # dataset_test_select = dataset_test.get_selected_variables_dataset(seq_selected_variable)
        # p_test, __ = permutation_runner.run_test(dataset_test_select)
        seq_test_tst_result_container = permutation_tests.permutation_tests(
            dataset_test=dataset_test,
            variable_selection_approach='hard',
            interpretable_mmd_result=trained_result,
            n_permutation_test=requests.n_permutation_test,
            distance_functions=requests.test_distance_functions,)
        p_test = np.max([__tst_obj.p_value for __tst_obj in seq_test_tst_result_container])        
    else:
        p_test = None
    # end if
    
    test_power_dev = trained_result.trajectory_record_validation[-1].ratio
    
    end_exec = time.time()
    time_exec_wallclock = end_exec - start_exec
    
    epoch = trained_result.trajectory_record_training[-1].epoch
    return _AlgorithmOneRangeFunctionReturn(
        request=requests,
        trained_result=trained_result,
        indices_detected=seq_selected_variable,
        regularization_parameter=requests.regularization_parameter,
        trainer=pl_trainer,
        p_value_dev=p_dev,
        p_value_test=p_test,
        test_power_dev=test_power_dev,
        is_success=True,
        execution_time_wallclock=time_exec_wallclock,
        execution_time_cpu=-1.0,
        epochs=epoch)



def function_logging_manually(opt_result: _AlgorithmOneRangeFunctionReturn):
    """Logging the optimization result manually."""
    assert opt_result.trained_result is not None
    mmd_estimator_opt_result = opt_result.trained_result
    
    seq_pl_logger = opt_result.request.pl_loggers
    assert seq_pl_logger is not None

    for __pl_logger in seq_pl_logger:
        log_postprocess_manually(
            mmd_estimator_opt_result=mmd_estimator_opt_result,
            indices_detected=opt_result.indices_detected,
            pl_logger=__pl_logger,
            aux_object={
                "p_value_dev": opt_result.p_value_dev,
                "p_value_test": opt_result.p_value_test}
        )

    
def _f_create_new_key(obj: _AlgorithmOneRangeFunctionReturn) -> float: 
    assert obj.test_power_dev is not None
    assert obj.p_value_dev is not None
    return obj.test_power_dev * (1 - obj.p_value_dev)


def __run_parameter_space_search(training_dataset: BaseDataset,
                                 regularization_search_parameter: RegularizationSearchParameters,
                                 estimator: BaseMmdEstimator,
                                 pytorch_trainer_config: PytorchLightningDefaultArguments,
                                 base_training_parameter: InterpretableMmdTrainParameters,
                                 post_process_handler: ty.Optional[PostProcessLoggerHandler],
                                 dask_client: ty.Optional[Client],
                                 validation_dataset: ty.Optional[BaseDataset],
                                 ) -> SelectionResult:
    # regularization parameter range
    # Optuna based reg. parameter search.
    if validation_dataset is None:
        dataset_whole = training_dataset
    else:    
        dataset_whole = training_dataset.merge_new_dataset(validation_dataset)
    # end if
    
    _reg_lower = RegularizationParameter(
        lambda_1=regularization_search_parameter.reg_parameter_search_lower_l1,
        lambda_2=regularization_search_parameter.reg_parameter_search_lower_l2)
    _reg_upper = RegularizationParameter(
        lambda_1=regularization_search_parameter.reg_parameter_search_upper_l1,
        lambda_2=regularization_search_parameter.reg_parameter_search_upper_l2)
    
    logger.debug('I am running search of lambda-parameter lower & upper bound...')
    copy_base_estimator = deepcopy(estimator)
    if regularization_search_parameter.search_strategy == 'optuna':
        reg_search_result = optuna_search(
            dataset_train=dataset_whole,
            mmd_estimator=copy_base_estimator,
            base_training_parameter=base_training_parameter,
            pytorch_trainer_config=pytorch_trainer_config,
            path_storage_backend_db=regularization_search_parameter.path_optuna_study_db,
            dask_client=dask_client,
            post_process_handler=post_process_handler,
            n_regularization_parameter=regularization_search_parameter.n_regularization_parameter,
            n_trials=regularization_search_parameter.n_search_iteration,
            concurrent_limit=regularization_search_parameter.max_concurrent_job,
            regularization_param_search_upper=_reg_upper,
            regularization_param_search_lower=_reg_lower)
        logger.debug('Finished seahing of lambda-parameter lower & upper bound. Processing next step: CV MMD-Opt.')
    elif regularization_search_parameter.search_strategy == 'heuristic':
        # Note: heuristic search is not recommended.
        # I implement this code just for keeping reproduction of the previous study.
        reg_search_result = heuristic_search(
            dataset=dataset_whole,
            mmd_estimator=copy_base_estimator,
            training_parameter=base_training_parameter,
            pytorch_trainer_config=pytorch_trainer_config,
            n_regularization_parameter=regularization_search_parameter.n_regularization_parameter,
            max_try=regularization_search_parameter.n_search_iteration,
            post_process_handler=post_process_handler,
            dask_client=dask_client,
            n_concurrent_run=regularization_search_parameter.max_concurrent_job,)
    else:
        raise ValueError(f'Unknown search strategy: {regularization_search_parameter.search_strategy}')
    # end if

    return reg_search_result



def __update_mmd_estimator_initial_value(reg_search_result: SelectionResult, mmd_estimator: BaseMmdEstimator):
    """Updating the initial value of the ARD weights of the MMD estimator.
    The initial value is the lower bound of the regularization parameter.
    """
    raise NotImplementedError('Currently, I do not confirm this function in terms of effects in optimisation.')

    logger.info('reg_search_result is given. I use the weights at the lower bound of the regularization parameter.')
    assert reg_search_result.dict_regularization2model_parameter is not None
    __dict_reg2model = reg_search_result.dict_regularization2model_parameter.items()
    ## TODO L1 regularization parameter sorting only...
    t_mmd_model_lower = sorted(__dict_reg2model, key=lambda o: o[0][0])[0]
    mmd_model_lower = t_mmd_model_lower[1]
    if isinstance(mmd_model_lower, dict):
        lower_bound_weights = mmd_model_lower['kernel_obj.ard_weights']
    elif isinstance(mmd_model_lower, BaseMmdEstimator):
        lower_bound_weights = mmd_model_lower.kernel_obj.ard_weights
    else:
        raise ValueError(f'Unknown type of mmd_model_lower: {type(mmd_model_lower)}')
    # end if
    mmd_estimator.kernel_obj.ard_weights = torch.nn.Parameter(lower_bound_weights)
    
    return mmd_estimator


def __generate_distributed_arguments(seq_regularization_parameters: ty.List[RegularizationParameter],
                                     base_training_parameter: InterpretableMmdTrainParameters,
                                     mmd_estimator: BaseMmdEstimator,
                                     dataset_training: BaseDataset,
                                     dataset_dev: BaseDataset,
                                     pytorch_trainer_config: PytorchLightningDefaultArguments,
                                     path_work_dir: Path,
                                     variable_detection_method: str,
                                     dataset_test: ty.Optional[BaseDataset] = None,
                                     #  permutation_test_runner_base: ty.Optional[PermutationTest] = None,
                                     test_distance_functions: ty.Tuple[str, ...] = ('sliced_wasserstein',),
                                     n_permutation_test: int = 500
                                     ) -> ty.List[_AlgorithmOneRangeFunctionRequestPayload]:
    seq_function_request_payload = []
    # generate request parameters for distributed computing.
    for regularization_parameter in seq_regularization_parameters:
        # updating the regularization parameter
        _param = deepcopy(base_training_parameter)
        _param.regularization_parameter = regularization_parameter
        
        # updating pytorch lightning trainer
        # I do wanna set enable_checkpointing False.
        _pl_trainer_config = deepcopy(pytorch_trainer_config)
        
        # logger is possible to update or add.
        # logic. the function take the basic logger object.
        # this function rewrites just run name.
        
        _path_trained_model = path_work_dir / str(regularization_parameter)

        # set a parameter to train models
        __request_payload = _AlgorithmOneRangeFunctionRequestPayload(
            regularization_parameter=regularization_parameter,
            mmd_estimator=deepcopy(mmd_estimator),
            parameter_variable_trainer=_param,
            dataset_training=dataset_training,
            dataset_dev=dataset_dev,
            dataset_test=dataset_test,
            pl_trainer_config=_pl_trainer_config,
            variable_detection_method=variable_detection_method,
            path_work_dir=_path_trained_model,
            n_permutation_test=n_permutation_test,
            test_distance_functions=test_distance_functions,
            permutation_test_runner=None,
        )
        seq_function_request_payload.append(__request_payload)
        # end if
    # end for

    return seq_function_request_payload        


def __select_model(is_p_value_filter: bool, 
                   seq_optimized_mmd: ty.List[_AlgorithmOneRangeFunctionReturn]
                   ) -> ty.Tuple[ty.Optional[AlgorithmOneIndividualResult], ty.List[AlgorithmOneIndividualResult]]:
    """Selecting a model (estimator) where test_power_dev is max. and p_value_** is min.
    Returns
    -------
    individual_result: ty.Optional[AlgorithmOneIndividualResult]
        Selected model.
    trained_models: ty.List[AlgorithmOneIndividualResult]
    """
    # model selection
    if is_p_value_filter:
        # p-value filter.
        # An MMD estimator must provide a set of variables. The variables must reject the null hypothesis when permutation test checks.
        # I choose an MMD estimator where its test-power is the highest.
        model_selected: ty.Optional[_AlgorithmOneRangeFunctionReturn]
        seq_model_selection = [opt_obj for opt_obj in seq_optimized_mmd 
                               if opt_obj.p_value_dev is not None and opt_obj.p_value_dev < 0.05]
        if len(seq_model_selection) > 0:
            model_selected = sorted(seq_model_selection, key=lambda obj: obj.test_power_dev, reverse=True)[0]
            # __path_model = model_selected.request.path_model.as_posix()
            assert model_selected is not None
            if isinstance(model_selected.request.regularization_parameter, tuple):
                regularization_parameter = RegularizationParameter(*model_selected.request.regularization_parameter)
            elif isinstance(model_selected.request.regularization_parameter, RegularizationParameter):
                regularization_parameter = model_selected.request.regularization_parameter
            else:
                raise TypeError()
            # end if
            
            if model_selected.trained_result is None:
                trained_ard_weights = None
            else:
                trained_ard_weights = model_selected.trained_result.ard_weights_kernel_k
            # end if
                
            individual_result = AlgorithmOneIndividualResult(
                regularization_parameter=regularization_parameter,
                selected_variables=model_selected.indices_detected,
                trained_ard_weights=trained_ard_weights,
                p_value_dev=model_selected.p_value_dev,
                test_power_dev=model_selected.test_power_dev,
                p_value_test=model_selected.p_value_test,
                pl_loggers=model_selected.request.pl_loggers,
                interpretable_mmd_train_result=model_selected.trained_result)
        else:
            model_selected = None
            individual_result = None
        # end if
    else:
        # selecting a model (estimator) where test_power_dev is max. and p_value_** is min.
        # creating a new sort key; test_power_dev + (1 - p_value_**).
        # sorting seq_estimators by the created key.
        _seq_model_s_sort_key = [(_f_create_new_key(opt_obj), opt_obj) for opt_obj in seq_optimized_mmd]
        model_selected = sorted(_seq_model_s_sort_key, key=lambda x: x[0], reverse=True)[0][1]
        assert model_selected is not None
        
        if isinstance(model_selected.request.regularization_parameter, tuple):
            regularization_parameter = RegularizationParameter(*model_selected.request.regularization_parameter)
        elif isinstance(model_selected.request.regularization_parameter, RegularizationParameter):
            regularization_parameter = model_selected.request.regularization_parameter
        else:
            raise TypeError()
        # end if
        
        if model_selected.trained_result is None:
            trained_ard_weights = None
        else:
            trained_ard_weights = model_selected.trained_result.ard_weights_kernel_k
        # end if
        
        individual_result = AlgorithmOneIndividualResult(
                        regularization_parameter=regularization_parameter,
                        selected_variables=model_selected.indices_detected,
                        trained_ard_weights=trained_ard_weights,
                        p_value_dev=model_selected.p_value_dev,
                        test_power_dev=model_selected.test_power_dev,
                        p_value_test=model_selected.p_value_test,
                        pl_loggers=model_selected.request.pl_loggers,
                        interpretable_mmd_train_result=model_selected.trained_result)
    # end if
    
    trained_models = [
        AlgorithmOneIndividualResult(
            regularization_parameter=__model.request.regularization_parameter,
            selected_variables=__model.indices_detected,
            trained_ard_weights=__model.trained_result.ard_weights_kernel_k,
            p_value_dev=__model.p_value_dev,
            test_power_dev=__model.test_power_dev,
            p_value_test=__model.p_value_test,
            pl_loggers=__model.request.pl_loggers,
            interpretable_mmd_train_result=__model.trained_result)
        for __model in seq_optimized_mmd
    ]
    return individual_result, trained_models



def __run_algorithm_one_min_max_param_range(
    candidate_regularization_parameters: PossibleTypeRegularizationParameter,
    dataset_training: BaseDataset,
    dataset_dev: BaseDataset,
    mmd_estimator: BaseMmdEstimator,
    base_training_parameter: InterpretableMmdTrainParameters,
    pytorch_trainer_config: PytorchLightningDefaultArguments,
    path_work_dir: Path,
    optuna_regularization_search_parameter: RegularizationSearchParameters,
    post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
    dask_client: ty.Optional[Client] = None,
    dataset_test: ty.Optional[BaseDataset] = None,
    distributed_batch_size: int = 5,
    # permutation_test_runner_base: ty.Optional[PermutationTest] = None,
    variable_detection_method: str = "hist_based",
    test_distance_functions: ty.Tuple[str, ...] = ('sliced_wasserstein',),
    n_permutation_test: int = 500
    ) -> ty.List[_AlgorithmOneRangeFunctionReturn]:
    """Running algorithm one with min. and max. of regularization parameters.
    """
    
    seq_optimized_mmd = []
    if candidate_regularization_parameters == 'auto_min_max_range':
        # finding lower and upper bound of regularization parameters.
        param_search_result = __run_parameter_space_search(
            training_dataset=dataset_training,
            regularization_search_parameter=optuna_regularization_search_parameter,
            estimator=mmd_estimator,
            pytorch_trainer_config=pytorch_trainer_config,
            base_training_parameter=base_training_parameter,
            post_process_handler=post_process_handler,
            dask_client=dask_client,
            validation_dataset=dataset_dev)
        logger.info(f'Parameter search result: {param_search_result.regularization_parameters}')
        logger.debug('I update the initial value of the ARD weights of the MMD estimator.')
        # mmd_estimator = __update_mmd_estimator_initial_value(reg_search_result=param_search_result, mmd_estimator=mmd_estimator)
        seq_regularization_parameters = param_search_result.regularization_parameters
    elif isinstance(candidate_regularization_parameters, SelectionResult):
        seq_regularization_parameters = candidate_regularization_parameters.regularization_parameters
        logger.debug('I update the initial value of the ARD weights of the MMD estimator.')
        # mmd_estimator = __update_mmd_estimator_initial_value(reg_search_result=candidate_regularization_parameters, mmd_estimator=mmd_estimator)
    elif isinstance(candidate_regularization_parameters, list):
        assert all(isinstance(obj, RegularizationParameter) for obj in candidate_regularization_parameters)
        seq_regularization_parameters = candidate_regularization_parameters
    else:
        raise ValueError(f'`candidate_regularization_parameters` is unexpected object. Current type -> {type(candidate_regularization_parameters)}')
    # end if
    seq_function_request_payload = __generate_distributed_arguments(
        seq_regularization_parameters=seq_regularization_parameters,
        base_training_parameter=base_training_parameter,
        mmd_estimator=mmd_estimator,
        dataset_training=dataset_training,
        dataset_dev=dataset_dev,
        pytorch_trainer_config=pytorch_trainer_config,
        path_work_dir=path_work_dir,
        variable_detection_method=variable_detection_method,
        dataset_test=dataset_test,
        # permutation_test_runner_base=permutation_test_runner_base,
        test_distance_functions=test_distance_functions,
        n_permutation_test=n_permutation_test)

    # batching function requests.
    # when distributed system is not enough trustable, we want to split a task pooling into smaller batches.
    # distributed_batch_size == -1, if you do not care.
    
    if distributed_batch_size == -1:
        seq_batched_requests = [seq_function_request_payload]
    else:
        seq_batched_requests = more_itertools.batched(
            seq_function_request_payload, distributed_batch_size
    )
    
    # region: execute function requests
    for batch_request in seq_batched_requests:
        if dask_client is None:
            return_obj = [__run_optimization_estimator(req) for req in batch_request]
        else:
            assert dask_client is not None
            task_queue = dask_client.map(__run_optimization_estimator, batch_request)
            return_obj = dask_client.gather(task_queue)
        # end if
        assert isinstance(return_obj, list)
        
        # post-processing distributed computing
        for opt_result in return_obj:
            assert isinstance(opt_result, _AlgorithmOneRangeFunctionReturn)
            seq_optimized_mmd.append(opt_result)
            # save a model
            if opt_result.is_success:
                assert opt_result.request.path_work_dir is not None
                Path(opt_result.request.path_work_dir).mkdir(
                    parents=True, exist_ok=True
                )
                path_save_model = opt_result.request.path_work_dir / "trained_model.pt"
                torch.save(opt_result.trained_result, path_save_model)
            # end if

            if post_process_handler is not None:
                __run_name = opt_result.get_key_id()
                __loggers = post_process_handler.initialize_logger(run_name=__run_name, group_name=detection_algorithm_one.__name__)
                post_process_handler.log(loggers=__loggers, target_object=opt_result)
            # end if
        # end for
    # endregion    
    return seq_optimized_mmd


def __run_algorithm_one_search_objective_based(
    candidate_regularization_parameters: PossibleTypeRegularizationParameter,
    dataset_training: BaseDataset,
    dataset_dev: BaseDataset,
    mmd_estimator: BaseMmdEstimator,
    base_training_parameter: InterpretableMmdTrainParameters,
    pytorch_trainer_config: PytorchLightningDefaultArguments,
    path_work_dir: Path,
    optuna_regularization_search_parameter: RegularizationSearchParameters,
    post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
    dask_client: ty.Optional[Client] = None,
    dataset_test: ty.Optional[BaseDataset] = None,
    # permutation_test_runner_base: ty.Optional[PermutationTest] = None,
    variable_detection_method: str = "hist_based",
    test_distance_functions: ty.Tuple[str, ...] = ('sliced_wasserstein',),
    n_permutation_test: int = 500
    ) -> ty.List[_AlgorithmOneRangeFunctionReturn]:
    """Private function. 
    
    Running algorithm one XXXXXXXXXXXXXXXXXXXXXX
    
    Returns
    -------
        ty.List[_FunctionReturn]
    """    
    # -------------------------------------------------------------
    # running variable detection algorithm without any regularization parameter.
    __training_param = deepcopy(base_training_parameter)
    __training_param.regularization_parameter = RegularizationParameter(0, 0)
    
    if dataset_training.is_dataset_on_ram():
        dataset_training = dataset_training.generate_dataset_on_ram()
    # end if
    
    if dataset_dev.is_dataset_on_ram():
        dataset_dev = dataset_dev.generate_dataset_on_ram()
    # end if
    
    variable_detector = InterpretableMmdDetector(
        mmd_estimator=deepcopy(mmd_estimator),
        training_parameter=__training_param,
        dataset_train=dataset_training,
        dataset_validation=dataset_dev)
    pl_trainer_obj = pl.Trainer(**pytorch_trainer_config.as_dict())
    pl_trainer_obj.fit(variable_detector)
    detection_result_obj = variable_detector.get_trained_variables()
    variable_detected = detect_variables(detection_result_obj.ard_weights_kernel_k)
    # -------------------------------------------------------------
    _reg_lower = RegularizationParameter(
        lambda_1=optuna_regularization_search_parameter.reg_parameter_search_lower_l1,
        lambda_2=optuna_regularization_search_parameter.reg_parameter_search_lower_l2)
    _reg_upper = RegularizationParameter(
        lambda_1=optuna_regularization_search_parameter.reg_parameter_search_upper_l1,
        lambda_2=optuna_regularization_search_parameter.reg_parameter_search_upper_l2)
        
    selection_result = optuna_opt_parameter_search.main(
        dataset_train=dataset_training,
        dataset_dev=dataset_dev,
        dataset_test=dataset_test,
        mmd_estimator=mmd_estimator,
        base_training_parameter=base_training_parameter,
        pytorch_trainer_config=pytorch_trainer_config,
        path_storage_backend_db=optuna_regularization_search_parameter.path_optuna_study_db,
        dask_client=dask_client,
        post_process_handler=post_process_handler,
        n_trials=optuna_regularization_search_parameter.n_search_iteration,
        concurrent_limit=optuna_regularization_search_parameter.max_concurrent_job,
        regularization_param_search_upper=_reg_upper,
        regularization_param_search_lower=_reg_lower,
        test_distance_functions=test_distance_functions,
        variables_without_regularization=variable_detected,
        n_permutation_test=n_permutation_test
    )
    logger.info(f'Parameter search result: {selection_result.regularization_parameters}')
    seq_optimized_mmd = []
    assert selection_result.dict_regularization2optuna_return is not None
    for __reg, __optuna_dask_res in selection_result.dict_regularization2optuna_return.items():
        _func_return = _AlgorithmOneRangeFunctionReturn(
            request=_AlgorithmOneRangeFunctionRequestPayload(
                parameter_variable_trainer=__optuna_dask_res.mmd_train_result.training_parameter,
                pl_trainer_config=pytorch_trainer_config,
                regularization_parameter=__reg,
                mmd_estimator=mmd_estimator,
                dataset_training=dataset_training,
                dataset_dev=dataset_dev,
                dataset_test=dataset_test,
                # permutation_test_runner=permutation_test_runner_base,
                variable_detection_method=variable_detection_method),
            trained_result=__optuna_dask_res.mmd_train_result,
            indices_detected=__optuna_dask_res.selected_variables,
            regularization_parameter=__reg,
            trainer=None,
            test_power_dev=__optuna_dask_res.test_power_dev,
            p_value_dev=__optuna_dask_res.p_value_dev,
            p_value_test=__optuna_dask_res.p_value_test,
            is_success=True,
            execution_time_wallclock=__optuna_dask_res.execution_time_wallclock,
            execution_time_cpu=__optuna_dask_res.execution_time_cpu,
            epochs=__optuna_dask_res.epochs)
        seq_optimized_mmd.append(_func_return)
    # end for

    return seq_optimized_mmd


def detection_algorithm_one(
    mmd_estimator: BaseMmdEstimator,
    pytorch_trainer_config: PytorchLightningDefaultArguments,
    base_training_parameter: InterpretableMmdTrainParameters,
    dataset_training: BaseDataset,
    dataset_dev: BaseDataset,
    candidate_regularization_parameters: PossibleTypeRegularizationParameter = 'search_objective_based',
    regularization_search_parameter: RegularizationSearchParameters = RegularizationSearchParameters(),
    dask_client: ty.Optional[Client] = None,
    distributed_batch_size: int = -1,
    variable_detection_method: str = "hist_based",
    is_p_value_filter: bool = False,
    # permutation_test_runner_base: ty.Optional[PermutationTest] = None,
    dataset_test: ty.Optional[BaseDataset] = None,
    path_work_dir: ty.Optional[Path] = None,
    post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
    test_distance_functions: ty.Tuple[str, ...] = ('sliced_wasserstein',),
    n_permutation_test: int = 500
    ) -> AlgorithmOneResult:
    """
    Args
    -----
    mmd_estimator: MMD estimator.
    pytorch_trainer_config: Pytorch lightning trainer configuration.
    dataset_training: Training dataset.
    dataset_dev: Dev dataset.
    candidate_regularization_parameters: A set of regularization parameters.
        `SelectionResult`: Result object from searching module of regularization parameters.
        `List[RegularizationParameter]`: A set of regularization parameters.
        `auto`: Optuna seaches for min. and max. of regularization parameters.
        `search_objective_based`: Optuna seaches for MMD-estimators by minimizing a customized obj. function.
    regularization_search_parameter: Parameter search configuration.
        Use only when `candidate_regularization_parameters` is None.
    dask_client: Dask client.
    distributed_batch_size: Batch size for distributed computing.
    variable_detection_method: Variable detection method.
    is_p_value_filter: If True, the algorithm selects a model where p-value is less than 0.05.
    permutation_test_runner_base: Permutation test runner.
    dataset_test: Test dataset.
    path_work_dir: Working directory.
    post_process_handler: Post process handler.
    n_permutation_test: Number of permutation test.
    
    Returns
    -------
    AlgorithmOneResult
    """
    __reg_mode: str
    # checking possible input type
    if isinstance(candidate_regularization_parameters, list):
        assert all(isinstance(obj, RegularizationParameter) for obj in candidate_regularization_parameters)
        __reg_mode = 'min_max_param_range'
    else:
        assert isinstance(candidate_regularization_parameters, SelectionResult) or isinstance(candidate_regularization_parameters, str)
        if isinstance(candidate_regularization_parameters, str):
            assert candidate_regularization_parameters in ('auto_min_max_range', 'search_objective_based'), \
                f'candidate_regularization_parameters must be either "auto_min_max_range" or "search_objective_based".'
            if candidate_regularization_parameters == 'auto_min_max_range':
                __reg_mode = 'min_max_param_range'
            elif candidate_regularization_parameters == 'search_objective_based':
                __reg_mode = 'search_objective_based'
            else:
                raise ValueError(f'candidate_regularization_parameters is unexpected value: {candidate_regularization_parameters}')
        else:
            raise ValueError(f'`candidate_regularization_parameters` is unexpected object. Current type -> {type(candidate_regularization_parameters)}')
        # end if
    # end if
    
    
    if path_work_dir is None:
        path_work_dir = Path(mkdtemp())
    # end if
    
    seq_optimized_mmd = []
    
    # TODO: inconsistent implementation. `__run_algorithm_one_min_max_param_range` has the argument `permutation_test_runner_base`.
    # However, `permutation_test_runner_base` in `__run_algorithm_one_search_objective_based` has no effect.
    # I have to make them consistent.
    if __reg_mode == 'min_max_param_range':
        seq_optimized_mmd = __run_algorithm_one_min_max_param_range(
            candidate_regularization_parameters=candidate_regularization_parameters,
            dataset_training=dataset_training,
            dataset_dev=dataset_dev,
            mmd_estimator=mmd_estimator,
            base_training_parameter=base_training_parameter,
            pytorch_trainer_config=pytorch_trainer_config,
            path_work_dir=path_work_dir,
            optuna_regularization_search_parameter=regularization_search_parameter,
            post_process_handler=post_process_handler,
            dask_client=dask_client,
            dataset_test=dataset_test,
            distributed_batch_size=distributed_batch_size,
            # permutation_test_runner_base=permutation_test_runner_base,
            variable_detection_method=variable_detection_method,
            test_distance_functions=test_distance_functions,
            n_permutation_test=n_permutation_test
            )
    elif __reg_mode == 'search_objective_based':
        seq_optimized_mmd = __run_algorithm_one_search_objective_based(
            candidate_regularization_parameters=candidate_regularization_parameters,
            dataset_training=dataset_training,
            dataset_dev=dataset_dev,
            mmd_estimator=mmd_estimator,
            base_training_parameter=base_training_parameter,
            pytorch_trainer_config=pytorch_trainer_config,
            path_work_dir=path_work_dir,
            optuna_regularization_search_parameter=regularization_search_parameter,
            post_process_handler=post_process_handler,
            dask_client=dask_client,
            dataset_test=dataset_test,
            # permutation_test_runner_base=permutation_test_runner_base,
            variable_detection_method=variable_detection_method,
            test_distance_functions=test_distance_functions,
            n_permutation_test=n_permutation_test)
    else:
        raise ValueError(f'__reg_mode is unexpected value: {__reg_mode}')
    
    individual_result, trained_models = __select_model(is_p_value_filter=is_p_value_filter, seq_optimized_mmd=seq_optimized_mmd)
        
    if path_work_dir is None:
        shutil.rmtree(path_work_dir.as_posix())
    # end if
    
    # TODO adding optuna search result.
    return AlgorithmOneResult(individual_result, trained_models)
