import typing

import numpy as np
import torch


def get_l2_distance_based_variables(x: typing.Union[np.ndarray, torch.Tensor],
                                    y: typing.Union[np.ndarray, torch.Tensor],
                                    threshold: float,
                                    threshold_selection: str = 'normalized_min') -> typing.Tuple[typing.List[int], np.ndarray]:
    """Variable detection baseline using L2 distance.

    :param x: (sample-size, dimension-size)
    :param y: (sample-size, dimension-size)
    :param threshold: a threshold to detect variables.
    :param threshold_selection: normalizing values [0., 1.0] before selecting indices.
    :return: a list of detected variable indices
    """
    # baseline prediction
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    diff_xy = np.abs(x - y)
    diff_l2 = np.sum(diff_xy ** 2, axis=0)

    if threshold_selection == 'normalized_min':
        normalized_diff_l2 = diff_l2 / np.max(diff_l2)
        index_prediction = [ind.item() for ind in np.argsort(- normalized_diff_l2) if normalized_diff_l2[ind] > threshold]
    else:
        threshold_l2 = np.mean(diff_l2, axis=0)
        index_prediction = [ind.item() for ind in np.argsort(- diff_l2) if diff_l2[ind] > threshold_l2]
    # end if

    return index_prediction, diff_l2
