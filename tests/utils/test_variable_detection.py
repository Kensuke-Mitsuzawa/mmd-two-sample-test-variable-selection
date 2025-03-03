import numpy as np
import torch
import json

from mmd_tst_variable_detector.utils.variable_detection import detect_variables



# Test function
def test_detect_variables_hist_based():
    input_weights = np.array([0.1, 0.11, 0.00001, 0.00000002, 0.000005, 0.0000004])
    list_index = detect_variables(variable_weights=input_weights)
    assert all(isinstance(ind, int) for ind in list_index)
    assert json.dumps(list_index)
    assert set(list_index) == set([0, 1])
    

if __name__ == "__main__":
    test_detect_variables_hist_based()
