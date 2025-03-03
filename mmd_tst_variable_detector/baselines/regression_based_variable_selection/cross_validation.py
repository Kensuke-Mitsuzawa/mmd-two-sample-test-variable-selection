# import random
# import typing
# import logging
# import itertools
# from dataclasses import dataclass

# import joblib
# from distributed import Client
# import typing as ty

# import torch
# import numpy as np

# from ...datasets.base import BaseDataset
# from ...utils.permutation_test_runner import PermutationTest
# from ...utils.commons import (
#     # parameters
#     SubLearnerTrainingResult,
#     DistributedComputingParameter
# )


# from .algorithm_one import (
#     _FunctionRequestPayload,
#     _FunctionReturn,
#     CandidateModelContainer,
#     _run_optimization_estimator,
#     _aggregate_p_values
# )

# from ...logger_unit import handler 
# logger = logging.getLogger(f'{__package__}.{__name__}')
# logger.addHandler(handler)

# # -------------------------------------------------------------
# # Public APIs

# @dataclass
# class RegressionCrossValidationAggregatedResult:
#     stable_s_hat: ty.Optional[ty.List[int]]  # a list of selected coordinates
#     array_s_hat: ty.Optional[np.ndarray]  # a tensor (dim,) computed by avg(stability-score).
#     stability_score_matrix: ty.Optional[np.ndarray]  # a tensor (|reg|, dim)


# # Trained parameter
# @dataclass
# class RegressionCrossValidationTrainedParameter:
#     regression_model_container: CandidateModelContainer
#     trained_models: ty.List[_FunctionReturn]
#     stability_score_matrix: ty.Optional[np.ndarray]  # 2d-array (|reg|, dim)
#     array_s_hat: ty.Optional[np.ndarray]  # 1d-array (dim,)
#     stable_s_hat: ty.List[int]
    
    
# @dataclass
# class RegressionCrossValidationAlgorithmParameter:
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

# @dataclass
# class RegressionCrossValidationTrainParameters:
#     algorithm_parameter: RegressionCrossValidationAlgorithmParameter
#     dist_parameter: DistributedComputingParameter
#     # base_training_parameter: ty.Optional[InterpretableMmdTrainParameters]
#     # wandb_logger_parameter: ty.Optional[WandbLoggerParameter] = None
#     computation_backend: str = 'joblib'  # local or dask

#     def __post_init__(self):
#         assert self.computation_backend in ('single', 'dask', 'joblib')


# # -------------------------------------------------------------
# # util functions

# # def dask_worker_script(args: RequestDistributedFunction) -> SubLearnerTrainingResult:
# #     """A function that Dask workers calls.
# #     :param args: 6 elements. See the asserting message below.
# #     :return: (task_id, trained-result)
# #     """
# #     from mmd_tst_variable_detector.utils.commons import OptimizationException
# #     assert len(args) == 7, "args must be: task-id training_parameter, dataset_train, dataset_val, trainer_lightning, mmd_estimator. See the function definition."
# #     task_id: ty.Tuple[RegularizationParameter, int] = args[0]
# #     training_parameter: InterpretableMmdTrainParameters = args[1]
# #     __dataset_train: BaseDataset = args[2]
# #     __dataset_val: BaseDataset = args[3]
# #     trainer_lightning: pl.Trainer = args[4]
# #     mmd_estimator: BaseMmdEstimator = args[5]
# #     ss_algorithm_param: StabilitySelectionAlgorithmParameter = args.stability_algorithm_param

# #     dim_x, dim_y = __dataset_train.get_dimension_flattened()
# #     init_ard_weights = torch.ones(dim_x)
# #     mmd_estimator.kernel_obj.ard_weights = torch.nn.Parameter(init_ard_weights)

# #     dataset_train = __dataset_train.copy_dataset()
# #     dataset_val = __dataset_val.copy_dataset()
# #     try:
# #         variable_detector = InterpretableMmdDetector(mmd_estimator=mmd_estimator,
# #                                                training_parameter=training_parameter,
# #                                                dataset_train=dataset_train,
# #                                                dataset_validation=dataset_val)
# #         trainer_lightning.fit(variable_detector)
# #     except OptimizationException:
# #         return SubLearnerTrainingResult(
# #             job_id=task_id,
# #             training_parameter=training_parameter,
# #             training_result=None,
# #             p_value_selected=None,
# #             variable_detected=None)
# #     # end try
# #     weights_detector_result = variable_detector.get_trained_variables()

# #     variables = detect_variables(
# #         variable_detection_approach=ss_algorithm_param.ard_weight_selection_strategy,
# #         variable_weights=weights_detector_result.ard_weights_kernel_k,
# #         threshold_weights=ss_algorithm_param.ard_weight_minimum,
# #         is_normalize_ard_weights=True)

# #     # Permutation Test on the detected variables
# #     dataset_val_selected = dataset_val.get_selected_variables_dataset(variables)
    
# #     def func_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
# #         if isinstance(x, torch.Tensor):
# #             x = x.cpu().detach().numpy()
# #         if isinstance(y, torch.Tensor):
# #             y = y.cpu().detach().numpy()
        
# #         v = ot.sliced_wasserstein_distance(x, y)
# #         return torch.tensor(v) 
# #     # end def
    
# #     permutation_test_runner = PermutationTest(
# #         func_distance=func_distance,
# #         batch_size=training_parameter.batch_size,
# #         n_permutation_test=1000)
# #     p_value_model_selection, __null_distribution = permutation_test_runner.run_test(dataset_val_selected)
    
# #     return SubLearnerTrainingResult(task_id,
# #                                     training_parameter,
# #                                     weights_detector_result,
# #                                     p_value_model_selection,
# #                                     variables)


# class RegressionCrossValidationInterpretableVariableDetector(object):
#     def __init__(self,
#                  candidate_model_container: CandidateModelContainer,
#                  training_parameter: RegressionCrossValidationTrainParameters,
#                  training_dataset: BaseDataset,
#                  seq_permutation_runner: ty.List[PermutationTest],
#                  validation_dataset: ty.Optional[BaseDataset] = None,
#                  path_work_dir: ty.Optional[str] = None):
#         """
        
#         Args:
#             training_dataset: A dataset of the whole available data.
#             seq_permutation_runner: A set of Permutation test runners with various distance functions.
#             validation_dataset: A dev-dataset of the development data. 
#                                 If it's given, this class concatenates the training and dev datasets, and generates the whole dataset.
#         """
#         self.candidate_model_container = candidate_model_container
        
#         self.training_parameter = training_parameter

#         self.training_dataset = training_dataset
#         self.validation_dataset = validation_dataset
        
#         self.seq_permutation_runner = seq_permutation_runner
        
#         self.path_work_dir = path_work_dir

#         self.seq_trained_variables = []

#     def __generate_distributed_argument(self,
#                                         model_container: CandidateModelContainer
#                                         ) -> ty.List[_FunctionRequestPayload]:
#         """Generate arguments for distributed computing. 
#         It is the double-loop structure of Regression-models and Sub-Dataset.

#         :param model_container:
#         :return:
#         """
#         seq_task_arguments = []
        
#         assert self.training_parameter.algorithm_parameter.sampling_strategy in ('cross-validation'), \
#             f'Currently, this class supports only cross-validation mode. Your input is {self.training_parameter.algorithm_parameter.sampling_strategy}'

#         # concatenate the training and dev datasets, generating the whole dataset.
#         if self.validation_dataset is None:
#             dataset_whole = self.training_dataset
#         else:
#             dataset_whole = self.training_dataset.merge_new_dataset(self.validation_dataset)
#         # end if
        
#         for __key, reg_model in model_container:  # for-loop over set-of-models.
#             # assert isinstance(reg_model, AcceptableClass), f'Your input model is not acceptable. {reg_model}'

#             assert dataset_whole is not None, 'dataset_whole must not be None.'
#             __n_sample_train = int(
#                 len(dataset_whole) * self.training_parameter.algorithm_parameter.ratio_subsampling)                
#             sample_ids_train = random.sample(range(len(dataset_whole)), k=__n_sample_train)
#             sample_ids_val = list(set(range(len(dataset_whole))) - set(sample_ids_train))
            
#             __, new_dataset_train = dataset_whole.get_subsample_dataset(sample_ids=sample_ids_train)
#             __, new_dataset_val = dataset_whole.get_subsample_dataset(sample_ids=sample_ids_val)

#             for __i_cv_iter in range(self.training_parameter.algorithm_parameter.n_subsampling):
#                 assert new_dataset_train is not None and new_dataset_val is not None
#                 new_dataset_train = new_dataset_train.copy_dataset()
#                 new_dataset_val = new_dataset_val.copy_dataset()

#                 # __parameter: InterpretableMmdTrainParameters = copy.deepcopy(self.training_parameter.base_training_parameter)
#                 __task_id = f'{__key}-{__i_cv_iter}'
                
#                 __function_request = _FunctionRequestPayload(
#                     task_id=__task_id,
#                     model_type_id=__key,
#                     regression_model=reg_model,
#                     dataset_training=new_dataset_train,
#                     dataset_dev=new_dataset_val,
#                     dataset_test=None,
#                     set_permutation_test_runners=self.seq_permutation_runner,
#                     variable_detection_method=self.training_parameter.algorithm_parameter.ard_weight_selection_strategy,
#                     path_work_dir=self.path_work_dir)
#                 seq_task_arguments.append(__function_request)
#             # end for
#         # end for
#         return seq_task_arguments

#     def __distributed_joblib_backend(self,
#                                      task_parameters: ty.List[_FunctionRequestPayload]
#                                      ) -> ty.List[_FunctionReturn]:
#         """

#         :return:
#         """
#         batch_n = self.training_parameter.dist_parameter.job_batch_size
#         seq_batch = [task_parameters[i * batch_n:(i + 1) * batch_n] for i in range((len(task_parameters) + batch_n - 1) // batch_n)]

#         seq_results = []
#         for on_job_batch in seq_batch:
#             __seq_results = joblib.Parallel(
#                 n_jobs=self.training_parameter.dist_parameter.n_joblib,
#                 backend=self.training_parameter.dist_parameter.joblib_backend)(
#                 joblib.delayed(_run_optimization_estimator)(args) for args in on_job_batch)
#             seq_results += __seq_results

#         return seq_results

#     def __distributed_dask_backend(self,
#                                    task_parameters: ty.List[_FunctionRequestPayload]
#                                    ) -> ty.List[_FunctionReturn]:
#         batch_n = self.training_parameter.dist_parameter.job_batch_size
#         seq_batch = [task_parameters[i * batch_n:(i + 1) * batch_n] for i in range((len(task_parameters) + batch_n - 1) // batch_n)]

#         client = Client(self.training_parameter.dist_parameter.dask_scheduler_address)
#         seq_results = []
#         for on_job_batch in seq_batch:
#             task_queue = client.map(_run_optimization_estimator, on_job_batch)
#             __seq_results: ty.List[SubLearnerTrainingResult] = client.gather(task_queue)  # type: ignore
#             seq_results += __seq_results

#         return seq_results

#     def __non_distributed_single_backend(self,
#                                          task_parameters: ty.List[_FunctionRequestPayload]
#                                          ) -> ty.List[_FunctionReturn]:
#         """
#         :return:
#         """
#         batch_n = self.training_parameter.dist_parameter.job_batch_size
#         seq_batch = [task_parameters[i * batch_n:(i + 1) * batch_n] for i in range((len(task_parameters) + batch_n - 1) // batch_n)]

#         seq_results = []
#         for on_job_batch in seq_batch:
#             __seq_results = []
#             for args in on_job_batch:
#                 seq_results.append(_run_optimization_estimator(args))
#                 seq_results += __seq_results
#             # end for
#             # save opt results
#             # if self.resume_checkpoint_saver is not None:
#             #     for sub_learner_result in __seq_results:
#             #         self.resume_checkpoint_saver.save_checkpoint(sub_learner_result)
#             # end if
#         # end for

#         return seq_results

#     @staticmethod
#     def __get_weighted_stability_score(
#         stability_score_original: torch.Tensor,
#         array_mmd_weight_vector: torch.Tensor) -> torch.Tensor:
#         """Getting a product of stability-score * weight-vector
#         """
#         stability_score = torch.clone(stability_score_original)
#         for reg_i, w_and_ss_score in enumerate(zip(array_mmd_weight_vector, stability_score_original)):
#             w, ss_score = w_and_ss_score
#             weighted_stability_score = w * ss_score
#             stability_score[reg_i] = weighted_stability_score
#         # end for
#         return stability_score

#     def __get_stable_s_hat(self,
#                            stability_score: np.ndarray,
#                            threshold: float,
#                            strategy_stability_score: str = 'mean',
#                            is_normalize: bool = True
#                            ) -> ty.Tuple[ty.List[int], np.ndarray]:
#         """Private method.

#         Get stable_s_hat that shows us the best lambda condition dimension wise.
#         Args:
#             stability_score: Tensor of (|lambda-candidate|, |dimension|). Values represent probabilities.
#             threshold: A threshold against the probability.
#             strategy_stability_score: an operation on the stability score matrix. 'max' or 'mean'.
#             is_normalize: True, the S.S. is normalized by (S.S.) / max(S.S.). The values range is [0.0, 1.0].
#         Returns: (`stable_s_hat`, `stability_score_agg`) `stable_s_hat` indicates indices of dimensions above the threshold.
#         A list of index number or a list of a tuple of (index, column).
#         """
#         assert strategy_stability_score in self.training_parameter.algorithm_parameter.strategy_stability_score

#         if strategy_stability_score == 'max':
#             stability_score_agg = np.max(stability_score, axis=0)
#         elif strategy_stability_score == 'mean':
#             stability_score_agg = np.mean(stability_score, axis=0)
#         else:
#             raise NotImplementedError()
#         # end if

#         if is_normalize:
#             stability_score_agg = stability_score_agg / np.max(stability_score_agg)
#         # end if
#         stable_s_hat = [k for k, score in enumerate(stability_score_agg) if score > threshold]

#         # end if

#         return stable_s_hat, stability_score_agg

#     @staticmethod
#     def __weight_score(array_value: np.ndarray, 
#                        weighting_mode: str, 
#                        p_value: float, 
#                        test_power_val: float) -> np.ndarray:
#         if weighting_mode == 'plane':
#             return array_value
#         elif weighting_mode == 'p_value':
#             return array_value * p_value
#         elif weighting_mode == 'test_power':
#             return array_value * test_power_val
#         elif weighting_mode == 'p_value_filter':
#             return np.zeros(len(array_value)) if p_value > 0.05 else array_value
#         elif weighting_mode == 'p_value_filter_test_power':
#             return np.zeros(len(array_value)) if p_value > 0.05 else array_value * test_power_val
#         elif weighting_mode == 'p_value_min_test_power':
#             # idea: bigger test_power and smaller p_value -> better 
#             return array_value * (test_power_val * (1 - p_value))
#         else:
#             raise NotImplementedError()

#     def _compute_stability_score(self,
#                                  model_container: CandidateModelContainer,
#                                  seq_trained_variables: ty.List[_FunctionReturn],
#                                  weighting_mode: str,
#                                  score_value: str
#                                  ) -> np.ndarray:
#         """Private method.

#         Computing Stability Score matrix. The matrix size is (|Lambda|, Dimension).

#         :param regularization_parameters:
#         :param seq_trained_variables:
#         :param is_save_learning_trajectory:
#         :return: (stability score matrix, `ty.List[_FunctionReturn]`)
#         """
#         # region: for-loop per regularization_parameter
#         assert len(set([__obj.weights.shape for __obj in seq_trained_variables])) == 1, \
#             'The shape of the weights must be the same.'
#         __dimension_size = seq_trained_variables[0].weights.shape[0]
        
#         types_regression_model_type = model_container.get_model_keys()
#         stability_score = np.zeros((len(types_regression_model_type), __dimension_size))

#         # make group of model-key and results
#         __sorted_variables = sorted(seq_trained_variables, key=lambda o: o.request.model_type_id)  # type: ignore
#         model_key2variables = {
#             k: list(g_obj) 
#             for k, g_obj in itertools.groupby(__sorted_variables, key=lambda o: o.request.model_type_id)}
                
#         for __model_i, __model_name in enumerate(model_key2variables):
#             # selecting results where the regression model key matches.
#             __seq_results = model_key2variables[__model_name]
#             # seq_results = [
#             #     o for o in seq_trained_variables
#             #     if o.request.regression_model.__class__.__name__ == __model_name]
#             n_subsampling = len(__seq_results)

#             # array shape: (N, D).
#             stability_variable = np.zeros((n_subsampling, __dimension_size))

#             # ---------------------------------------------------------
#             # region: Aggregation per regularization parameter

#             for __i_sub_learner, __result_container in enumerate(__seq_results):
#                 if __result_container.is_success == False:
#                     # if the optimization is failed, the result is None.
#                     continue
#                 # end if

#                 if score_value == 'ard':
#                     # note: the term is not correct. Correctly, supposed to be 'weights' or 'coef'.
#                     # I use 'ard' for the codebase consistency.
#                     # weights array is already normalized on power(2).
#                     __array_value = __result_container.weights
#                 elif score_value == 'variable':
#                     __array_value = np.zeros(__dimension_size)
#                     for __ind in __result_container.indices_detected:
#                         __array_value[__ind] = 1.0
#                     # end for
#                 else:
#                     raise NotImplementedError()
#                 # end if

#                 # weighting
#                 __p_value = _aggregate_p_values(__result_container)
                
#                 __array_value = self.__weight_score(
#                     array_value=__array_value,
#                     weighting_mode=weighting_mode,
#                     p_value=__p_value,
#                     test_power_val=1.0)
#                 stability_variable[__i_sub_learner, :] = __array_value
#             # endregion: end for
#             # ---------------------------------------------------------

#             score_at_reg = np.sum(stability_variable, axis=0) / n_subsampling  # type: ignore

#             # 3. saving the record
#             stability_score[__model_i, :] = score_at_reg

#         # endregion: end for
#         return stability_score

#     def run_sub_learners_lambda_parameter(self,
#                                           model_container: CandidateModelContainer
#                                           ) -> ty.List[_FunctionReturn]:
#         """Public method.

#         Run sub-learners with a regularization parameter. A controller of joblib or dask job scheduler.

#         "job-id" consists of "LambdaParam-SubsamplingID".

#         :param model_container:
#         :return: list of `_FunctionReturn`
#         """
#         # generating a set of task parameters
#         seq_request_function = self.__generate_distributed_argument(model_container)
        
#         already_trained_sub_learners = []
#         if self.training_parameter.computation_backend == 'single':
#             already_trained_sub_learners += self.__non_distributed_single_backend(seq_request_function)
#         elif self.training_parameter.computation_backend == 'joblib':
#             already_trained_sub_learners += self.__distributed_joblib_backend(seq_request_function)
#         elif self.training_parameter.computation_backend == 'dask':
#             already_trained_sub_learners += self.__distributed_dask_backend(seq_request_function)
#         else:
#             raise NotImplementedError(f'No backend named {self.training_parameter.computation_backend}')
#         # end if

#         # log to wandb
#         # if self.training_parameter.wandb_logger_parameter.is_wandb_aggregated_log:
#         #     __d_sort = {reg_tuple: list(g_obj) for reg_tuple, g_obj in itertools.groupby(
#         #         sorted(already_trained_sub_learners, key=lambda o: o.job_id[0]), key=lambda o: o.job_id[0])}
#         #     for reg_param, stack_sub_learner_training in __d_sort.items():
#         #         wandb_logger_utils.visualize_wandb_sub_learner(
#         #             reg_param=reg_param,
#         #             stack_sub_learner_training=stack_sub_learner_training,
#         #             wandb_parameter=self.training_parameter.wandb_logger_parameter
#         #         )
#         # # end if

#         return already_trained_sub_learners

#     def get_stability_score(
#             self,
#             seq_trained_variables: ty.Optional[ty.List[_FunctionReturn]] = None,
#             weighting_mode: ty.Optional[str] = None,
#             score_value: ty.Optional[str] = None
#     ) -> RegressionCrossValidationAggregatedResult:
#         """Public method.

#         Executing post-processing after the ARD weights optimization based on sub-sampled dataset.
#         The post-processing continues as follows,
#         1. Getting the Stability score.
#         2. purgeing inappropriate regularization parameters.
#         3. Getting Stable S, which is a set of predicted coordinates.
#         4. ARD weights optimization only with the predicted coordinates.
#         Use this method when you already have ARD weights optimization results based on sub-sampled dataset,
#         and when you wanna recalculate Stability scores.
#         Args:
#             seq_trained_variables: list of ARD weights optimization results generated based on the sub-sampled dataset.
#             weighting_mode: If None, `StabilitySelectionAlgorithmParameter.aggregation_mode` is used.
#             score_value: If None, `StabilitySelectionAlgorithmParameter.score_value` is used.
#         """
#         if seq_trained_variables is None:
#             seq_trained_variables = self.seq_trained_variables
#         if weighting_mode is None:
#             weighting_mode = self.training_parameter.algorithm_parameter.weighting_mode
#         if score_value is None:
#             score_value = self.training_parameter.algorithm_parameter.stability_score_base

#         __stack_sub_learner_training = [
#             __learner_obj for __learner_obj in seq_trained_variables if __learner_obj.is_success]
#         # Do nothing when input is empty.
#         if len(__stack_sub_learner_training) == 0:
#             return RegressionCrossValidationAggregatedResult(
#                 stable_s_hat=[],
#                 array_s_hat=None,
#                 stability_score_matrix=None)
#         # end if

#         # computing the Stability Score
#         stability_score_original = self._compute_stability_score(
#             model_container=self.candidate_model_container,
#             seq_trained_variables=seq_trained_variables,
#             weighting_mode=weighting_mode,
#             score_value=score_value
#         )

#         # getting the Stable-S that is a set of detected coordinates
#         stable_s_hat, stability_score_agg = self.__get_stable_s_hat(
#             stability_score=stability_score_original,
#             threshold=self.training_parameter.algorithm_parameter.threshold_stability_score,
#             strategy_stability_score=self.training_parameter.algorithm_parameter.strategy_stability_score,
#             is_normalize=self.training_parameter.algorithm_parameter.is_normalize_agg_stability_score
#         )
#         return RegressionCrossValidationAggregatedResult(
#             stable_s_hat=stable_s_hat,
#             array_s_hat=stability_score_agg,
#             stability_score_matrix=stability_score_original)

#     def run_cv_detection(self) -> RegressionCrossValidationTrainedParameter:
#         """Public method.

#         Interface method. Running a MMD optimization for ARD weights.
#         Args:
#         Returns:
#             `TrainedMmdParameters`
#         """
#         # running ARD weights optimization with sub-sampled dataset.
#         seq_trained_variables = self.run_sub_learners_lambda_parameter(self.candidate_model_container)
#         self.seq_trained_variables = seq_trained_variables

#         # TODO need MLflow logging here.
        
#         stability_selection_aggregated = self.get_stability_score(seq_trained_variables=seq_trained_variables)

#         # TODO need MLflow logging here.
#         assert stability_selection_aggregated.stability_score_matrix is not None
#         assert stability_selection_aggregated.array_s_hat is not None        
        
#         if stability_selection_aggregated.stable_s_hat is None:
#             ss_trained_parameter = RegressionCrossValidationTrainedParameter(
#                 regression_model_container=self.candidate_model_container,
#                 trained_models=seq_trained_variables,
#                 stability_score_matrix=stability_selection_aggregated.stability_score_matrix,
#                 array_s_hat=stability_selection_aggregated.array_s_hat,
#                 stable_s_hat=[])
#         elif len(stability_selection_aggregated.stable_s_hat) > 0:
#             ss_trained_parameter = RegressionCrossValidationTrainedParameter(
#                 regression_model_container=self.candidate_model_container,
#                 trained_models=seq_trained_variables,
#                 stability_score_matrix=stability_selection_aggregated.stability_score_matrix,
#                 array_s_hat=stability_selection_aggregated.array_s_hat,
#                 stable_s_hat=stability_selection_aggregated.stable_s_hat)
#         else:
#             ss_trained_parameter = RegressionCrossValidationTrainedParameter(
#                 regression_model_container=self.candidate_model_container,
#                 trained_models=seq_trained_variables,
#                 stability_score_matrix=stability_selection_aggregated.stability_score_matrix,
#                 array_s_hat=stability_selection_aggregated.array_s_hat,
#                 stable_s_hat=stability_selection_aggregated.stable_s_hat)
#         # end if

#         return ss_trained_parameter
