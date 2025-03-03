import torch
import torch.utils.data

import numpy as np

from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

from mmd_tst_variable_detector.baselines.regression_based_variable_selection.variable_detector_regression import (
    TrainedResultRegressionBasedVariableDetector,
    RegressionBasedVariableDetector
)
from mmd_tst_variable_detector import (
    SimpleDataset,
    PermutationTest
)
from mmd_tst_variable_detector.baselines.regression_based_variable_selection.commons import (
    _base_func_distance_sliced_wasserstein,
    _base_func_distance_sinkhorn
)

from ...data_generator import test_data_xy_linear



def test_RegressionBasedVariableDetector():
    (x_tensor, y_tensor), dim_ground_truth = test_data_xy_linear(dim_size=20,
                                                                 sample_size=1000,
                                                                 ratio_dependent_variables=0.1)
    dataset_obj = SimpleDataset(x_tensor, y_tensor)
    
    for model_module in [Ridge(), Lasso(), ARDRegression(), LogisticRegression(), SVR(kernel="linear")]:
        regression_module = RegressionBasedVariableDetector(
            regression_module=model_module,
            permutation_test_runners=[PermutationTest(enable_progress_bar=True)])
        trained_result = regression_module.run_variable_detection(dataset_obj, dataset_obj)
        assert isinstance(trained_result, TrainedResultRegressionBasedVariableDetector)
        assert isinstance(trained_result.weight_vector, np.ndarray) or trained_result.weight_vector is None
        if isinstance(trained_result.weight_vector, np.ndarray):
            assert trained_result.weight_vector.shape == (20,)
