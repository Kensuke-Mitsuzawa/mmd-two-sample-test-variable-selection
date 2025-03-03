import typing as ty

from ..detection_algorithm.interpretable_mmd_detector import InterpretableMmdTrainResult


def tune_trained_mmd_models(
        tuned_mmd_models: ty.List[InterpretableMmdTrainResult],
        tuning_metric: str = 'mmd',
        select_on: str = 'training') -> ty.List[InterpretableMmdTrainResult]:
    """Sorting multiple optimized MMD optimizers.

    :param tuned_mmd_models: a list of optimized MMD models.
    :param tuning_metric: recommended to use "mmd".
    :param select_on: 'training' or 'validation'
    :return:
    """
    def key_sort_trained_result(obj: InterpretableMmdTrainResult) -> float:
        if select_on == 'training':
            return getattr(obj.trajectory_record_training[-1], tuning_metric)
        else:
            return getattr(obj.trajectory_record_validation[-1], tuning_metric)
    # end def

    seq_sorted_mmd_result = sorted(tuned_mmd_models, key=key_sort_trained_result, reverse=True)

    return seq_sorted_mmd_result


