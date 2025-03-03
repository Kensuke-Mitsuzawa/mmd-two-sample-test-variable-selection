import typing as ty

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from ..detection_algorithm.commons import EvaluationVariableDetection
from .variable_detection import detect_variables


def evaluate_trained_variables(ard_weights: torch.Tensor,
                               ground_truth_index: ty.List[int],
                               variable_detection_approach: str = 'hist_based',
                               is_normalize_ard_weights: bool = True,
                               threshold_ard_weights: float = 0.1
                               ) -> ty.Tuple[EvaluationVariableDetection, ty.List[int]]:
    """Evaluating the trained ARD weights detection.
    Kernel K is for the variable X, Kernel L is for the variable Y.
    
    Parameters
    ------------
    ard_weights: `torch.Tensor`
        Not only ARD weights, but any weights values.
    ground_truth_index:
    is_normalize_ard_weights: 
        adjusting ARD weights in a range of (0.0, 1.0).
    threshold_ard_weights: 
        a threshold for detecting ARD weights.
        
    Returns
    ------------
    A tuple with two elements. First element: Three tuples. A tuple represents (precision, recall, F). Three tuples are for X, Y, both.
    Second element: a list of detected indices.
    """
    assert variable_detection_approach in ['hist_based', 'threshold']

    def func_generate_array(shape_ard_weights: int, ground_truth_index: ty.List[int]) -> np.ndarray:
        __true = np.zeros(shape_ard_weights)
        for __d in ground_truth_index:
            __true[__d] = 1
        # end for
        return __true

    # end func

    def func_get_p_r_f(output_sklearn: np.ndarray) -> EvaluationVariableDetection:
        """Getting P, R, F from an output of scikit learn package.
        :param output_sklearn:
        :return: (precision, recall, F)
        """
        return EvaluationVariableDetection(output_sklearn[0][1], output_sklearn[1][1], output_sklearn[2][1])

    # end func

    # setting the ground truth
    x_true = func_generate_array(ard_weights.shape, ground_truth_index)

    detected_x__ = detect_variables(
        variable_detection_approach=variable_detection_approach,
        variable_weights=ard_weights,
        threshold_weights=threshold_ard_weights,
        is_normalize_ard_weights=is_normalize_ard_weights)
    x_pred = func_generate_array(ard_weights.shape, detected_x__)

    if len(detected_x__) == 0:
        return EvaluationVariableDetection(0.0, 0.0, 0.0), x_pred.tolist()
    else:    
        eval_x_tuple = precision_recall_fscore_support(x_true, x_pred)

        if isinstance(detected_x__, (torch.Tensor, np.ndarray)):
            detected_x__ = detected_x__.tolist()
        # end if
        
        return func_get_p_r_f(eval_x_tuple), detected_x__
