# import typing as ty
# from dataclasses import dataclass

# from ..detection_algorithm.interpretable_mmd_detector import TrainingResult, EvaluationVariableDetection
# from .permutation_test_runner import PermutationTest
# from ..datasets import SimpleDataset
# from .evaluate_variable_detection import detect_variables

# @dataclass
# class _PermutationTestResult:
#     mmd_model: TrainingResult
#     p_value: float


# def func_filter_p_value(model: _PermutationTestResult, p_value_threshold: float = 0.05) -> bool:
#     if model.p_value < p_value_threshold:
#         return True
#     else:
#         return False


# def func_filter_epochs(model: TrainingResult, max_epochs: int) -> bool:
#     if model.trajectory_record_validation[-1].epoch < max_epochs:
#         return True
#     else:
#         return False


# def func_key_select_ratio_criteria(model: ty.Tuple[TrainingResult, EvaluationVariableDetection, ty.Dict]) -> float:
#     ratio_value = model[0].trajectory_record_validation[-1].ratio
#     return ratio_value


# def run_permutation_tests(seq_optimized_mmd_models: ty.List[TrainingResult],
#                           permutation_test_runner: PermutationTest,
#                           validation_dataset: SimpleDataset) -> ty.List[_PermutationTestResult]:
#     """Run permutation tests for the given models."""
#     test_results = []
#     for __mmd_model in seq_optimized_mmd_models:
#         __variable_detect = detect_variables(__mmd_model.ard_weights_kernel_k)
#         __x = validation_dataset.x
#         __y = validation_dataset.y
#         __select_x = __x[:, __variable_detect]
#         __select_y = __y[:, __variable_detect]
#         __p_value, __null_distribution = permutation_test_runner.run_test(SimpleDataset(__select_x, __select_y))
#         test_results.append(_PermutationTestResult(__mmd_model, __p_value))
#     # end if
#     return test_results


# def select_model_heuristic(seq_optimized_mmd_models: ty.List[TrainingResult],
#                            max_epochs: int,
#                            p_value_test_threshold: float = 0.05,
#                            p_values_model_selection: ty.Optional[ty.List[float]] = None,
#                            permutation_test_runner: ty.Optional[PermutationTest] = None,
#                            validation_dataset: ty.Optional[SimpleDataset] = None,
#                            ) -> ty.Optional[_PermutationTestResult]:
#     """Select the best model from the list of models based on the following criteria:
#     1. p-value of sliced Wasserstein test < 0.05
#     2. select the max ratio of the validation loss

#     :param seq_optimized_mmd_models:
#     :param permutation_test_runner:
#     :param p_values_model_selection:
#     :param validation_dataset:
#     :param max_epochs:
#     :param p_value_test_threshold:
#     :return: None if no model is selected (no model is < p_value_test_threshold).
#     """
#     if p_values_model_selection is None and permutation_test_runner is None:
#         raise Exception('Either p_values_model_selection or permutation_test_runner is mandatory.')
#     if p_values_model_selection is None and (permutation_test_runner is None and validation_dataset is None):
#         raise Exception('validation_dataset must be given.')

#     seq_optimized_mmd_models = [model for model in seq_optimized_mmd_models if func_filter_epochs(model, max_epochs)]
#     if len(seq_optimized_mmd_models) == 0:
#         return None
#     # end if

#     if p_values_model_selection is None:
#         test_results = run_permutation_tests(seq_optimized_mmd_models, permutation_test_runner, validation_dataset)
#     else:
#         assert len(seq_optimized_mmd_models) == len(p_values_model_selection), 'length unmatch. N(models) != N(p-values)'
#         test_results = [_PermutationTestResult(m, __p) for m, __p in zip(seq_optimized_mmd_models, p_values_model_selection)]
#     # end if
#     __models_filter = [model for model in test_results if func_filter_p_value(model, p_value_test_threshold)]

#     if len(__models_filter) == 0:
#         return None
#     # end if
#     __sorted_models = sorted(__models_filter, key=func_key_select_ratio_criteria, reverse=True)
#     model_selected = __sorted_models[0]

#     return model_selected

