# from mmd_tst_variable_detector.utils.permutation_tests import sliced_wasserstein

# import torch
# import numpy as np
# from ot import sliced_wasserstein_distance as swd_original


# def test_sliced_wasserstein():
#     x = torch.from_numpy(np.random.normal(loc=0, scale=1, size=(100, 20)))
#     y = torch.from_numpy(np.random.normal(loc=0, scale=1, size=(100, 20)))    
    
#     d = sliced_wasserstein.sliced_wasserstein_distance(x, y, seed=10)
    
#     d_original = swd_original(x, y, seed=10)
#     assert torch.abs(d - d_original).item() < 1e-3