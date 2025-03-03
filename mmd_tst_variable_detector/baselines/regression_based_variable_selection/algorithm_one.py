# from collections.abc import Iterator
# import more_itertools
# import typing as ty
# import shutil
# import timeit
# import logging
# import time

# from copy import deepcopy
# from pathlib import Path
# from distributed import Client
# from tempfile import mkdtemp
# from dataclasses import dataclass 


# import torch
# import numpy as np
# from pytorch_lightning.loggers.logger import Logger

# from sklearn.linear_model import Ridge, Lasso, LogisticRegression, ARDRegression
# from sklearn.svm import SVR


# from mmd_tst_variable_detector import (
#     RegularizationParameter,
#     BaseDataset,
#     PermutationTest,
#     # ---------------------------------
#     # utils
#     detect_variables
# )
# # detector class sample selection based detector
# from ...utils.post_process_logger import function_overwrite_logger_mmd_estimator, log_postprocess_manually
# from ...logger_unit import handler 
# from .variable_detector_regression import (
#     TrainedResultRegressionBasedVariableDetector,
#     RegressionBasedVariableDetector,
# )
# from .data_models import (
#     CandidateModelContainer,
#     _FunctionRequestPayload,
#     _FunctionReturn,
#     RegressionAlgorithmOneResult,
#     RegressionAlgorithmOneIndividualResult)


# logger = logging.getLogger(f'{__package__}.{__name__}')
# logger.addHandler(handler)




# # ----------------------------------------------
# # algorithm definitions


# # Future plan. call this function by Ray tune. The target objective value is p-value.
# def _run_optimization_estimator(requests: _FunctionRequestPayload) -> _FunctionReturn:
#     """Executing training of an regression model.
#     Based on the optimization resut, this function selects variables.
#     Then, this function executes permutation test for the selected variables.
#     """
#     # Permutation Test for the dev. data
#     assert requests.set_permutation_test_runners is not None
    
#     start_cpu_time = time.process_time()
#     start_wall_time = timeit.default_timer()
    
#     regression_model = requests.regression_model
#     variable_detector = RegressionBasedVariableDetector(
#         regression_module=regression_model,
#         batch_size=requests.batch_size
#     )
#     variable_detector_result = variable_detector.fit(requests.dataset_training)
    
#     if variable_detector_result.weight_vector is None:
#         return _FunctionReturn(
#             request=requests,
#             indices_detected=[],
#             trained_model=variable_detector_result.regression_model,
#             weights=np.array([]),
#             seq_p_value_dev=None,
#             seq_p_value_test=None,
#             is_success=False)
#     # end if
    
#     seq_selected_variable = detect_variables(variable_weights=variable_detector_result.weight_vector, 
#                                              variable_detection_approach=requests.variable_detection_method)

#     # TODO
#     """Permutation Testについて、次のような設計
    
#     Permutation Test objがgiven -> そのまま実行
#     Permutation Test objがNone -> 選択された変数のみをで、MMD estimatorのoptimizationを再実行。
#     ２つのoptionを用意する。1. full sctratchでoptimization, 2. 用意されたARD weightsを初期値としてoptimization
#     """
    
#     # selected variables based on the optimized result.
#     # dataset_dev_select = requests.dataset_dev.get_selected_variables_dataset(seq_selected_variable)
    
#     seq_p_value_dev = []
#     seq_p_value_test = []
    
#     dataset_dev = requests.dataset_dev
    
#     seq_permutation_runner = requests.set_permutation_test_runners
#     for __permutation_runner in seq_permutation_runner:
#         if requests.tst_approach == "soft":        
#             p_dev, __ = __permutation_runner.run_test(
#                 requests.dataset_dev,
#                 featre_weights=torch.tensor(variable_detector_result.weight_vector))
#             seq_p_value_dev.append(p_dev)
#         elif requests.tst_approach == "hard":
#             __dataset_dev_new = dataset_dev.get_selected_variables_dataset(seq_selected_variable)
#             p_dev, __ = __permutation_runner.run_test(__dataset_dev_new)
#             seq_p_value_dev.append(p_dev)
#         else:
#             raise ValueError(f"Unknown tst_approach: {requests.tst_approach}")
#         # end if
    
#         # permutation test for test data
#         if requests.dataset_test is not None:
#             dataset_test_select = requests.dataset_test.get_selected_variables_dataset(seq_selected_variable)
#             p_test, __ = __permutation_runner.run_test(dataset_test_select)
#             seq_p_value_test.append(p_test)
#         else:
#             p_test = None
#         # end if
#     # end for
    
#     end_cpu_time = time.process_time()
#     end_wall_time = timeit.default_timer()
    
#     wallclock_execution_time = end_wall_time - start_wall_time
#     cpu_time_execution_time = end_cpu_time - start_cpu_time

#     return _FunctionReturn(
#         request=requests,
#         trained_model=variable_detector_result.regression_model,
#         weights=variable_detector_result.weight_vector,
#         indices_detected=seq_selected_variable,
#         seq_p_value_dev=seq_p_value_dev,
#         seq_p_value_test=seq_p_value_test,
#         is_success=True,
#         wallclock_execution_time=wallclock_execution_time,
#         cpu_time_execution_time=cpu_time_execution_time)



# # def function_logging_manually(opt_result: __FunctionReturn):
# #     """Logging the optimization result manually."""
# #     mmd_estimator_opt_result = opt_result.trained_result
    
# #     # seq_training_trajectory = mmd_estimator_opt_result.trajectory_record_training
# #     # seq_validation_trajectory = mmd_estimator_opt_result.trajectory_record_validation

# #     seq_pl_logger = opt_result.request.pl_loggers
# #     assert seq_pl_logger is not None

# #     for __pl_logger in seq_pl_logger:
# #         log_postprocess_manually(
# #             mmd_estimator_opt_result=mmd_estimator_opt_result,
# #             indices_detected=opt_result.indices_detected,
# #             pl_logger=__pl_logger,
# #             aux_object={
# #                 "p_value_dev": opt_result.p_value_dev,
# #                 "p_value_test": opt_result.p_value_test}
# #         )

    
# # def _f_create_new_key(obj: _FunctionReturn) -> float: 
# #     assert obj.p_value_dev is not None
# #     return (1 - obj.p_value_dev)


# def _aggregate_p_values(task_return_obj: _FunctionReturn, 
#                        mode: str = 'biggest') -> float:
#     assert mode in ('mean', 'biggest', 'smallest')
#     assert task_return_obj.seq_p_value_dev is not None
    
#     if mode == 'mean':
#         return np.mean(task_return_obj.seq_p_value_dev)
#     elif mode == 'biggest':
#         return np.max(task_return_obj.seq_p_value_dev)
#     elif mode == 'smallest':
#         return np.min(task_return_obj.seq_p_value_dev)
    
    


# def detection_algorithm_one(
#     dataset_training: BaseDataset,
#     dataset_dev: BaseDataset,  # used for permutation test.
#     candidate_models: CandidateModelContainer,
#     seq_permutation_test_runner_base: ty.List[PermutationTest],
#     dask_client: ty.Optional[Client] = None,
#     distributed_batch_size: int = -1,
#     variable_detection_method: str = "hist_based",
#     is_p_value_filter: bool = False,
#     dataset_test: ty.Optional[BaseDataset] = None,
#     path_work_dir: ty.Optional[Path] = None,
#     base_pl_loggers: ty.Optional[ty.List[Logger]] = None,
#     training_batch_size: int = -1
#     ) -> RegressionAlgorithmOneResult:
#     """Regression version of the Algorithm 1 in the paper.
#     """
#     if path_work_dir is None:
#         path_work_dir = Path(mkdtemp())
#     # end if
    
#     seq_optimized_models = []
#     seq_function_request_payload = []

#     # generate request parameters for distributed computing.
#     for __key, __model_container in list(candidate_models):
#         logger.debug(f'Executing parameter: {__key}...')
#         # updating logger
#         # TODO consider it later.
#         seq_logger_updated = []
#         # if base_pl_loggers is None:
#         #     pass
#         # else:
#         #     for __logger_obj in base_pl_loggers:
#         #         __logger_updated = function_update_logger_conf(__logger_obj, regularization_parameter)
#         #         seq_logger_updated.append(__logger_updated)
#         #     # end for
#         # # end if

#         # logger is possible to update or add.
#         # logic. the function take the basic logger object.
#         # this function rewrites just run name.
        
#         _path_trained_model = path_work_dir / __key
#         _path_trained_model.mkdir(parents=True, exist_ok=True)
        
#         # set a parameter to train models
#         __request_payload = _FunctionRequestPayload(
#             regression_model=__model_container,
#             dataset_training=dataset_training,
#             dataset_dev=dataset_dev,
#             dataset_test=dataset_test,
#             set_permutation_test_runners=seq_permutation_test_runner_base,
#             variable_detection_method=variable_detection_method,
#             path_work_dir=_path_trained_model,
#             pl_loggers=seq_logger_updated,
#             batch_size=training_batch_size)
#         seq_function_request_payload.append(__request_payload)
#         # end if
#     # end for

#     # batching function requests.
#     # when distributed system is not enough trustable, we want to split a task pooling into smaller batches.
#     # distributed_batch_size == -1, if you do not care.
#     if distributed_batch_size == -1:
#         seq_batched_requests = [seq_function_request_payload]
#     else:
#         seq_batched_requests = more_itertools.batched(
#             seq_function_request_payload, distributed_batch_size
#     )

#     # region: execute function requests
#     for batch_request in seq_batched_requests:
#         if dask_client is None:
#             return_obj = [_run_optimization_estimator(req) for req in batch_request]
#         else:
#             assert dask_client is not None
#             task_queue = dask_client.map(_run_optimization_estimator, batch_request)
#             return_obj = dask_client.gather(task_queue)
#         # end if
#         assert isinstance(return_obj, list)

#         # post-processing distributed computing
#         for opt_result in return_obj:
#             assert isinstance(opt_result, _FunctionReturn)
#             # save a model
#             if opt_result.is_success:
#                 seq_optimized_models.append(opt_result)
                
#                 assert opt_result.request.path_work_dir is not None
#                 Path(opt_result.request.path_work_dir).mkdir(
#                     parents=True, exist_ok=True
#                 )
#                 path_save_model = opt_result.request.path_work_dir / "trained_model.pt"
#                 torch.save(opt_result.trained_model, path_save_model)
#             # end if

#             # manually logging the result
#             # if opt_result.request.pl_loggers is not None:
#             #     # logging the result
#             #     function_logging_manually(opt_result)
#             # end if
#         # end for
#     # endregion


#     # model selection
#     if is_p_value_filter:
#         # p-value filter.
#         # An MMD estimator must provide a set of variables. The variables must reject the null hypothesis when permutation test checks.
#         # I choose an MMD estimator where its test-power is the highest.
#         model_selected: ty.Optional[_FunctionReturn]
#         seq_model_selection = [
#             opt_obj 
#             for opt_obj in seq_optimized_models 
#             if min(opt_obj.seq_p_value_dev) < 0.05]
        
#         if len(seq_model_selection) > 0:
#             model_selected = sorted(seq_model_selection, key=_aggregate_p_values, reverse=True)[0]
#             # __path_model = model_selected.request.path_model.as_posix()
#             assert model_selected is not None

#             individual_result = RegressionAlgorithmOneIndividualResult(
#                 model_obj=model_selected.request.regression_model,
#                 selected_variables=model_selected.indices_detected,
#                 trained_weights=model_selected.weights,
#                 seq_p_value_dev=model_selected.seq_p_value_dev,
#                 seq_p_value_test=model_selected.seq_p_value_test,
#                 pl_loggers=model_selected.request.pl_loggers,
#                 regression_model=model_selected.trained_model)
#         else:
#             model_selected = None
#             individual_result = None
#         # end if
#     else:
#         # selecting a model (estimator) where test_power_dev is max. and p_value_** is min.
#         # creating a new sort key; test_power_dev + (1 - p_value_**).
#         # sorting seq_estimators by the created key.
#         _seq_model_s_sort_key = [(_aggregate_p_values(opt_obj), opt_obj) for opt_obj in seq_optimized_models]
#         model_selected = sorted(_seq_model_s_sort_key, key=lambda x: x[0], reverse=True)[0][1]
#         assert model_selected is not None
        
#         individual_result = RegressionAlgorithmOneIndividualResult(
#                         model_obj=model_selected.request.regression_model,
#                         selected_variables=model_selected.indices_detected,
#                         trained_weights=model_selected.weights,
#                         seq_p_value_dev=model_selected.seq_p_value_dev,
#                         seq_p_value_test=model_selected.seq_p_value_test,
#                         pl_loggers=model_selected.request.pl_loggers,
#                         regression_model=model_selected.trained_model)
#     # end if
    
#     trained_models = [
#         RegressionAlgorithmOneIndividualResult(
#             model_obj=__model.request.regression_model,
#             selected_variables=__model.indices_detected,
#             trained_weights=__model.weights,
#             seq_p_value_dev=__model.seq_p_value_dev,
#             seq_p_value_test=__model.seq_p_value_test,
#             pl_loggers=__model.request.pl_loggers,
#             regression_model=__model.trained_model)
#         for __model in seq_optimized_models
#     ]
    
    
#     if path_work_dir is None:
#         shutil.rmtree(path_work_dir.as_posix())
#     # end if
    
#     return RegressionAlgorithmOneResult(individual_result, trained_models)
