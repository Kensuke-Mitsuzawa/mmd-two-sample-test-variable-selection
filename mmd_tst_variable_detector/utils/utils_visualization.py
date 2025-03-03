# import itertools
# import typing as t

# import matplotlib
# import matplotlib.figure
# import matplotlib.pyplot as plt
# import seaborn as sns

# import numpy as np
# import pandas

# from ..detection_algorithm.cross_validation_detector import CrossValidationTrainedParameter
# from ..detection_algorithm.interpretable_mmd_detector import TrajectoryRecord, InterpretableMmdTrainResult, RegularizationParameter
# from ..detection_algorithm.cross_validation_detector.checkpoint_saver import CheckPointSaverStabilitySelection

# # from ..detection_algorithm.cross_validation_detector import SubLearnerTrainingResult


# def visualize_training_trajectory():
#     pass





# # -----------------------------------------------------------------------------------------


# def __extract_optimized_mmd_value(
#     seq_training_result: t.List[InterpretableMmdTrainResult]) -> t.List[float]:
#     """Extracing the optimized mmd2 values.

#     :param seq_training_result:
#     :return:
#     """
#     seq_mmd_train = []

#     for training_result in seq_training_result:
#         trajectory: t.List[TrajectoryRecord] = training_result.trajectory_record_training
#         mmd_train = trajectory[-1].mmd
#         seq_mmd_train.append(mmd_train)
#     # end for

#     return seq_mmd_train

# def __func_sort___aggregate_optimization_logs(item_obj: SubLearnerTrainingResult) -> RegularizationParameter:
#     """_summary_

#     Args:
#         item_obj (SubLearnerTrainingResult): _description_

#     Returns:
#         t.Tuple[RegularizationParameter, int]: _description_
#     """
#     return item_obj.training_parameter.regularization_parameter
    


# def __aggregate_optimization_logs(
#         checkpoint_saver: CheckPointSaverStabilitySelection
# ) -> t.List[t.Tuple[t.Tuple, t.List[InterpretableMmdTrainResult], t.List[float]]]:
#     """

#     :param checkpoint_saver:
#     :return: list of tuple. A tuple is (aggregation-key (tuple), set of `TrainingResult`, set of optimized MMD2 values.)
#     """
#     __seq_trained_results = checkpoint_saver.load_checkpoint()

#     __g_obj = itertools.groupby(sorted(__seq_trained_results, key=__func_sort___aggregate_optimization_logs), key=__func_sort___aggregate_optimization_logs)

#     seq_agg_tuple = []

#     for key_reg, g_obj in __g_obj:
#         seq_learner_log = [t.training_result for t in g_obj if t.training_result is not None]
#         # agg_keys.append(key_reg)
#         # agg_training_trajectory.append(__extract_optimized_mmd_value(seq_learner_log))
#         # agg_training_log.append(seq_learner_log)
#         seq_agg_tuple.append((
#             (key_reg, key_reg[1]),
#             seq_learner_log,
#             __extract_optimized_mmd_value(seq_learner_log)
#         ))
#     # end for
#     return seq_agg_tuple


# def visualize_stability_score_modifications(
#         stability_selection_result: CrossValidationTrainedParameter,
#         checkpoint_saver: CheckPointSaverStabilitySelection,
#         figure_object: t.Optional[matplotlib.figure.Figure] = None
# ) -> matplotlib.figure.Figure:
#     """_summary_
#     Args:
#         stability_selection_result:
#         checkpoint_saver:
#         figure_object (t.Optional[matplotlib.figure.Figure]): _description_
#     Raises:
#         NotImplementedError: _description_
#     Returns:
#         matplotlib.figure.Figure: _description_
#     """

#     agg_training_log = __aggregate_optimization_logs(checkpoint_saver)

#     if figure_object is None:
#         figure_object, axes = plt.subplots(
#             ncols=4,
#             nrows=len(agg_training_log),
#             figsize=(20, 20))
#     else:
#         assert isinstance(figure_object, matplotlib.figure.Figure)
#         axes = figure_object.subplots(4, 1)
#     # end if
    
#     assert stability_selection_result.stability_score_matrix is not None, 'Stability score matrix is None.'

#     if len(stability_selection_result.stability_score_matrix.shape) == 2:
#         visualization_type = 'bar'
#     elif len(stability_selection_result.stability_score_matrix.shape) == 3:
#         visualization_type = 'heatmap'
#     else:
#         raise NotImplementedError()
#     # end if

#     stability_score = stability_selection_result.stability_score_matrix
    
#     for __i, aggregation_tuple in enumerate(agg_training_log):
#         regularization_tuple: t.Tuple[RegularizationParameter, int] = aggregation_tuple[0]

#         array_sub_learner_mmd = np.array(aggregation_tuple[2])
#         # MMD histogram visualization
#         pandas.Series(array_sub_learner_mmd).hist(ax=axes[__i][0])
#         avg_mmd_value = np.mean(array_sub_learner_mmd)
#         axes[__i][0].set_title('(L1, L2)={}-{}.\navg-mmd={:.3f}'.format(
#             regularization_tuple[0],
#             regularization_tuple[1],
#             avg_mmd_value), pad=20)

#         # median. of ARD weights
#         seq_sub_learner: t.List[InterpretableMmdTrainResult] = aggregation_tuple[1]
#         avg_ard_weights = np.median([sub_learner_obj.ard_weights_kernel_k.numpy() for sub_learner_obj in seq_sub_learner], axis=0)
#         if visualization_type == 'bar':
#             pandas.Series(avg_ard_weights).plot.bar(ax=axes[__i][1])
#             axes[__i][1].set_title('median(ARD weights)', pad=20)

#             # stability score
#             pandas.Series(stability_score[__i]).plot.bar(ax=axes[__i][2])
#             axes[__i][2].set_title('Stability score', pad=20)

#             # weighted stability score
#             if stability_score is not None:
#                 pandas.Series(stability_score[__i]).plot.bar(ax=axes[__i][3])
#                 axes[__i][3].set_title('Weighted stability score', pad=20)
#         elif visualization_type == 'heatmap':
#             sns.heatmap(avg_ard_weights, ax=axes[__i][1])
#             axes[__i][1].set_title('median(ARD weights)', pad=20)

#             # stability score
#             sns.heatmap(stability_score[__i], ax=axes[__i][2])
#             axes[__i][2].set_title('Stability score', pad=20)

#             # weighted stability score
#             if stability_score is not None:
#                 sns.heatmap(stability_score[__i], ax=axes[__i][3], robust=True, vmin=0.0, vmax=1.0)
#                 axes[__i][3].set_title('Weighted stability score', pad=20)
#     # end for
#     plt.subplots_adjust(hspace=1.0, wspace=0.5)
#     plt.close()
#     return figure_object
