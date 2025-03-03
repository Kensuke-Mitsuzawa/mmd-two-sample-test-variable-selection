# from pathlib import Path
# import typing as ty
# import logging
# from dataclasses import dataclass

# from frozendict import frozendict

# from copy import deepcopy
# from functools import partial
# from dataclasses import asdict
# from datetime import datetime

# from sklearn.linear_model import (
#     ARDRegression,
#     Ridge,
#     LogisticRegression
# )
# from sklearn.svm import SVR

# import numpy as np

# import ray
# from ray import tune
# from ray.air import RunConfig
# from ray.tune.search import optuna
# from ray.tune.search import ConcurrencyLimiter
# from ray.tune.sklearn import TuneSearchCV


# import pytorch_lightning as pl
# from pytorch_lightning.loggers.logger import Logger as PlLogger
# import pytorch_lightning.loggers

# from ...datasets import BaseDataset
# from ...utils.variable_detection import detect_variables
# from ...logger_unit import handler 

# from ...utils.permutation_test_runner import PermutationTest
# from ...utils.commons import (
#     RegularizationParameter, 
#     InterpretableMmdTrainParameters, 
#     InterpretableMmdTrainResult
# )
# from .algorithm_one import _run_optimization_estimator
# from .commons import (
#     DefaultPermutationTests
# )
# from .data_models import (
#     AcceptableClass,
#     CandidateModelContainer,
#     _FunctionRequestPayload,
#     _FunctionReturn,
#     # default search space
#     BaseSearchParameter,
#     SearchParameterARDRegression,
#     SearchParameterRidge,
#     SearchParameterLogisticRegression,
#     SearchParameterSVR,
# )


# msg_logger = logging.getLogger(f'{__package__}.{__name__}')
# msg_logger.addHandler(handler)


# @dataclass
# class VariableSelectionResultRayTstTuned:                 
#     selected_variable_model: _FunctionReturn
#     searched_variable_models: ty.List[_FunctionReturn]

    
# def __wrapper_ray_run_optimization_estimator(config,
#                                              sk_model: AcceptableClass,
#                                              dataset_training: BaseDataset,
#                                              dataset_dev: BaseDataset,
#                                              path_work_dir: Path,
#                                              set_permutation_test_runners: ty.Optional[ty.List[PermutationTest]],
#                                              pl_loggers: ty.Optional[ty.List[PlLogger]],
#                                              batch_size: int = -1
#                                              ):
#     # TODO: tst soft or hard
#     request = _FunctionRequestPayload(
#         regression_model=sk_model,
#         dataset_training=dataset_training,
#         dataset_dev=dataset_dev,
#         set_permutation_test_runners=set_permutation_test_runners,
#         path_work_dir=path_work_dir,
#         pl_loggers=pl_loggers,
#         batch_size=batch_size,
#     )

#     try:
#         tst_result = _run_optimization_estimator(requests=request)
#     except Exception as e:
#         msg_logger.exception(e)
#         ray.train.report({"is_success": False})
#     else:
#         assert tst_result.seq_p_value_dev is not None, 'Not found p-value of the dev set.'
#         p_value_max = max(tst_result.seq_p_value_dev)
        
#         ray.train.report({
#             "is_success": True,
#             "p_value_max": p_value_max,
#             "num_detected_variables": len(tst_result.indices_detected),
#             "detected_variables": tst_result.indices_detected,
#             "tst_result": tst_result._asdict()})


# def __obtain_results(results: ty.List[ray.train.Result]) -> ray.train.Result:
#     # the result value is a sequence object of ray.air.result.Result object.
#     # https://docs.ray.io/en/latest/_modules/ray/air/result.html
#     # sort by p_value_max and take the smallest one.
#     p_value_max = min([r.metrics['p_value_max'] for r in results if 'is_success' in r.metrics])
#     # sort the result by the number of detected variables.
#     seq_pair_tuple_max = [
#         __result_obj for __result_obj in results
#         if 'is_success' in __result_obj.metrics and  __result_obj.metrics['p_value_max'] == p_value_max]
#     # obtain the smallest lambda value where l1 + l2 is the smallest.
#     result_selected = sorted(seq_pair_tuple_max, key=lambda x: x.metrics['num_detected_variables'], reverse=True)

#     return result_selected


# def _function_logging_manually(seq_opt_result: ty.List[ray.train.Result],
#                                pl_loggers: ty.List[PlLogger]):
#     """Private function. Logging the optimization result manually."""
#     # I want to log the following information.
#     # trajectory of MMD, loss, ratio.
#     # trained ARD weights. (visually if possible).
#     # a relationship between lambda-parameter and test-power.

#     seq_lambda_vs_metric = []
#     for __pl_logger in pl_loggers:
#         for __opt_result in seq_opt_result:
#             __mmd_estimator_opt_result: InterpretableMmdTrainResult = __opt_result.metrics['trained_result']
#             __indices_detected: ty.List[int] = __opt_result.metrics['detected_variables']

#             # update the logger and pass to the procding operations.
#             __reg_parameter = RegularizationParameter(__opt_result.config['l1'], __opt_result.config['l2'])
#             __logger_copy = function_overwrite_logger_mmd_estimator(__pl_logger, __reg_parameter)
            
#             # logging training information.
#             log_postprocess_manually(mmd_estimator_opt_result=__mmd_estimator_opt_result,
#                                      indices_detected=__indices_detected,
#                                      pl_logger=__logger_copy)
            
#             seq_lambda_vs_metric.append({
#                 "regularization_l1": __reg_parameter.lambda_1,
#                 "regularization_l2": __reg_parameter.lambda_2,
#                 "n_detected_variables": len(__indices_detected),
#                 "test_power": __mmd_estimator_opt_result.trajectory_record_validation[-1].ratio,
#                 })
#         # end for

#         if isinstance(__pl_logger, pytorch_lightning.loggers.MLFlowLogger):
#             # logging X: lambda-parameter, Y: number of detected variables.
#             # logging X: lambda-parameter, Y: Test-Power.
#             for __summary_record in seq_lambda_vs_metric:
#                 __run_exp_name = f'l1={__summary_record["regularization_l1"]}-l2={__summary_record["regularization_l2"]}'
#                 _logger_summary = __copy_mlflow_logger(__pl_logger, experiment_name='Ray Parameter Search Summary', run_exp_name=__run_exp_name)
#                 _logger_summary.log_hyperparams({"l1": __summary_record["regularization_l1"], "l2": __summary_record["regularization_l2"]})
#                 _logger_summary.log_metrics(__summary_record)
#                 _logger_summary.finalize(status='success')
#             # end for
#         # end for

#     # end for


# def ray_bayesopt_search_tst_based(
#         dataset_train: BaseDataset,
#         dataset_dev: BaseDataset,
#         candidate_sklearn_models: CandidateModelContainer,
#         search_parameter_objects: ty.Optional[ty.List[BaseSearchParameter]] = None,
#         base_pl_loggers: ty.Optional[ty.List[PlLogger]] = None,
#         ray_run_name_prefix: str = 'ray_bayesopt_search_tst_based',
#         max_search_iteration: int = 50,
#         max_concurrent_run: int = 3,
#         path_ray_root: ty.Optional[Path] = None,
#         permutation_test_runners: ty.Optional[ty.List[PermutationTest]] = None,
#         ) -> VariableSelectionResultRayTstTuned:
    
#     assert (max_search_iteration / max_concurrent_run) > 2.0, \
#         'Too small `max_search_iteration`. Must be: (max_search_iteration / max_concurrent_run) > 2.0'
    
#     # TODO use ray.put for setting the dataset.
#     # For the moment, impossible to use shared memory of Ray for pytorch Dataset object.
#     # Hence, I limit the number of concurrent run to 1.
#     # Ref: https://docs.ray.io/en/latest/ray-core/patterns/pass-large-arg-by-value.html
#     # dataset_ray_shared = ray.put(dataset)
#     if max_concurrent_run > 3:
#         msg_logger.warning('Too large `max_concurrent_run`. You may encounter RAM related issues, insufficient RAM memory. In that csae, set smaller value for `max_concurrent_run`')
#     # end if
    
#     # -------------------------------------------------------------------
#     # get all regression models
#     seq_class_names = candidate_sklearn_models.get_model_class_names()

#     # if search_parameter_objects is None, use default search space.
#     if search_parameter_objects is None:
#         search_parameter_objects = [
#             SearchParameterARDRegression(),
#             SearchParameterRidge(),
#             SearchParameterLogisticRegression(),
#             SearchParameterSVR()
#         ]
#     # end if
    
#     dict_class2search_parameter = {
#         __param_obj.class_name: __param_obj.params for __param_obj in search_parameter_objects
#     }

#     if permutation_test_runners is None:
#         permutation_test_runners = DefaultPermutationTests
    
#     # ---------------------------------------------------------
#     # ray tune independently for candidate classes.
#     __stack_ray_search = []
    
#     for __class_name in seq_class_names:
#         msg_logger.info(f'Starting Ray Tune for {__class_name}')
#         assert __class_name in dict_class2search_parameter, f'Not found {__class_name} in the search space.'
        
#         __parameter_space = dict_class2search_parameter[__class_name]
        
#         if __class_name == 'ARDRegression':
#             __sk_model = ARDRegression()
#         elif __class_name == 'Ridge':
#             __sk_model = Ridge()
#         elif __class_name == 'LogisticRegression':
#             __sk_model = LogisticRegression()
#         elif __class_name == 'SVR':
#             __sk_model = SVR()
#         else:
#             raise NotImplementedError(f'Not implemented class name: {__class_name}')
#         # end if        
        
#         # define training config
#         algo = optuna.OptunaSearch()
#         algo = ConcurrencyLimiter(algo, max_concurrent=max_concurrent_run)
                
#         tune_config_min = tune.TuneConfig(
#             metric='p_value_max',  # target metric to be optimized.
#             mode="min",  # searching for the smallest detected variables.
#             search_alg=algo,
#             num_samples=max_search_iteration)

#         ray_run_name_prefix = f'{ray_run_name_prefix}_{datetime.now().isoformat()}'

#         if path_ray_root is None:
#             path_work_dir = Path('/tmp') / ray_run_name_prefix / __class_name
#         else:
#             path_work_dir = (path_ray_root / ray_run_name_prefix / __class_name)
#         # end if

#         msg_logger.debug(f'Working directory: {path_work_dir.as_posix()}')
#         run_config = RunConfig(
#             name=ray_run_name_prefix,
#             storage_path=(path_work_dir / 'ray').as_posix())

#         func_target_min = partial(__wrapper_ray_run_optimization_estimator,
#                                   sk_model=__sk_model,
#                                   dataset_training=dataset_train,
#                                   dataset_dev=dataset_dev,
#                                   path_work_dir=path_work_dir,
#                                   set_permutation_test_runners=permutation_test_runners,
#                                   pl_loggers=base_pl_loggers,
#                                   batch_size=-1)

#         tuner = tune.Tuner(
#             func_target_min,
#             tune_config=tune_config_min,
#             param_space=dict(__parameter_space),
#             run_config=run_config
#         )
    
#         msg_logger.debug(f'Starting Ray Tune')
#         __results_p_value_min = tuner.fit()
#         msg_logger.debug(f'Finished Ray Tune')
        
#         __stack_ray_search += list(__results_p_value_min)
#     # end for
#     sorted_result_search = __obtain_results(__stack_ray_search)
    
#     # ---------------------------------------------------------
#     # TODO I have to design the class to return.
#     # I want a list to save the Result values.
#     # I want a selected Result object.
    
#     selected_variable_model = _FunctionReturn(**sorted_result_search[0].metrics['tst_result'])
#     searched_variable_models = [
#         _FunctionReturn(**r.metrics['tst_result']) for r in sorted_result_search]
    
#     return VariableSelectionResultRayTstTuned(
#         selected_variable_model=selected_variable_model,
#         searched_variable_models=searched_variable_models
#     )
