import numpy as np
import typing as ty

import torch


# def variable_detection_hist_based(array_weights: np.ndarray, bins: int =100) -> ty.Tuple[float, ty.List[int]]:
#     """This algorithm is used for reporing result in TPAMI draft.
#     """
#     hist, bins = np.histogram(array_weights, bins=bins)
#     zero_index = np.where(hist == 0.0)[0]
#     if len(zero_index) == 0:
#         # Case: array([2, 2, 1, 2, 1, 6, 1, 1, 2, 2])
#         local_valley = [ind for ind, value in enumerate(hist) if (ind > 0) and (value < hist[(ind - 1)])]
#         smallest_valley = min(local_valley)
#     else:
#         # case: array([11,  3,  2,  3,  0,  0,  0,  0,  0,  1])
#         smallest_valley = min(np.where(hist == 0.0)[0])
#     # end if

#     threshold = bins[smallest_valley]
#     feature_index_detect = np.where(array_weights > threshold)[0].astype(int).tolist()
#     return threshold, feature_index_detect


def variable_detection_hist_based_ver2(array_weights: np.ndarray, 
                                       n_bins: ty.Optional[int] = None) -> ty.Tuple[float, ty.List[int]]:
    """improved version of `variable_detection_hist_based`.
    Major updateds
    - `bins` is auto. 1/5 of the input dimension.
    - Workaround when there is no "0" value valley.
    """
    if n_bins is None:
        if len(array_weights) < 100:
            n_bins = 100
        else:
            n_bins = int(array_weights.shape[0] / 5)
    # end if
    assert n_bins is not None and isinstance(n_bins, int)

    while True:
        hist, bins = np.histogram(array_weights, bins=n_bins)
        zero_index = np.where(hist == 0.0)[0]
        if len(zero_index) == 0:
            n_bins = n_bins + 50
            continue
        else:
            # case: array([11,  3,  2,  3,  0,  0,  0,  0,  0,  1])
            smallest_valley = min(np.where(hist == 0.0)[0])
            break
        # end if
    # while

    threshold = bins[smallest_valley]
    feature_index_detect = np.where(array_weights > threshold)[0].astype(int).tolist()
    threshold = float(threshold) if isinstance(threshold, np.ndarray) else threshold
    return threshold, feature_index_detect
 

# -----------------------------------------------------------------------
# normalization based detection
# -----------------------------------------------------------------------


def func_normalize(x: ty.Union[torch.Tensor, np.ndarray]) -> ty.Union[torch.Tensor, np.ndarray]:
    if isinstance(x, np.ndarray):
        return x / np.max(x)
    elif isinstance(x, torch.Tensor):
        return x / torch.max(x)
    else:
        raise Exception()


def variable_detection_threshold(
    ard_weights: ty.Union[np.ndarray, torch.Tensor],
    threshold_ard_weights: float = 0.1,
    is_normalize: bool = True
) -> ty.List[int]:
    if is_normalize:
        ard_weights = func_normalize(ard_weights)
    # end if

    if isinstance(ard_weights, np.ndarray):
        detected_x__ = np.where(ard_weights > threshold_ard_weights)[0]
        variables = detected_x__.tolist()
    elif isinstance(ard_weights, torch.Tensor):
        detected_x__: torch.Tensor = torch.where(ard_weights > threshold_ard_weights)[0]
        variables = detected_x__.numpy().tolist()
    else:
        raise Exception()

    return variables


def detect_variables(
    variable_weights: ty.Union[np.ndarray, torch.Tensor],
    variable_detection_approach: str = 'hist_based',
    threshold_weights: float = 0.1,
    hist_bins: ty.Optional[int] = None,
    is_normalize_ard_weights: bool = True
) -> ty.List[int]:
    """Interface function
    """
    assert variable_detection_approach in ['hist_based', 'threshold']

    if isinstance(variable_weights, torch.Tensor):
        variable_weights = variable_weights.numpy()
        assert isinstance(variable_weights, np.ndarray)
    # end if

    if variable_detection_approach == 'hist_based':
        threshold_, detected_x__ = variable_detection_hist_based_ver2(variable_weights, n_bins=hist_bins)
    elif variable_detection_approach == 'threshold':
        detected_x__ = variable_detection_threshold(
            ard_weights=variable_weights,
            threshold_ard_weights=threshold_weights,
            is_normalize=is_normalize_ard_weights)
    else:
        raise Exception()
    
    return detected_x__
