import typing
from datetime import datetime

import numpy as np
import torch


from mmd_tst_variable_detector.distance_module.l2_distance import L2Distance



def test_l2_distance_vector():
    """test case dimension is flatten vector / x coordinate."""
    x = torch.tensor(np.random.normal(size=(100, 10)))
    y = torch.tensor(np.random.normal(size=(100, 10)))

    __distance_module = L2Distance(coordinate_size=1)
 
    __s = datetime.now()   
    d_container_alpha = __distance_module._L2Distance__compute_l2_flat_vector_alpha(x, y)
    __time_beta = datetime.now() - __s

    __s = datetime.now()
    d_container_beta = __distance_module._L2Distance__compute_l2_flat_vector_beta(x, y)
    __time_alpha = datetime.now() - __s

    assert (d_container_alpha.d_xx - d_container_beta.d_xx).mean() < 1e-5
    assert (d_container_alpha.d_yy - d_container_beta.d_yy).mean() < 1e-5
    assert (d_container_alpha.d_xy - d_container_beta.d_xy).mean() < 1e-5

    

def test_l2_distance_xy_coordinates():
    alpha = torch.tensor(np.random.normal(size=(100, 10, 2)))  # xy coordinate
    beta = torch.tensor(np.random.normal(size=(100, 10, 2)))  # xy coordinate

    __distance_module = L2Distance(coordinate_size=2)
    d_container = __distance_module.compute_distance(alpha, beta)
    
    # commet: test of xy L2 distance
    max_sample_x = 5
    max_sample_y = 5
    
    for __i_x in range(max_sample_x):
        for __j_y in range(max_sample_y):   
            alpha_sample_zero = alpha[__i_x]
            beta_sample_zero = beta[__j_y]
            # commet: (x_0-x_1)^2 + (y_0-y_1)^2
            l2_x = (alpha_sample_zero[:, 0] - beta_sample_zero[:, 0]) ** 2
            l2_y = (alpha_sample_zero[:, 1] - beta_sample_zero[:, 1]) ** 2
            l2_total = (l2_x + l2_y).sum()
            
            assert (d_container.d_xy[__i_x][__j_y] - l2_total) < 1e-5
        # end for
    # end for

def test_l2_distance_xyz_coordinates():
    alpha = torch.tensor(np.random.normal(size=(100, 10, 3)))  # xy coordinate
    beta = torch.tensor(np.random.normal(size=(100, 10, 3)))  # xy coordinate

    __distance_module = L2Distance(coordinate_size=3)
    d_container = __distance_module.compute_distance(alpha, beta)
    
    # commet: test of xy L2 distance
    max_sample_x = 5
    max_sample_y = 5
    
    for __i_x in range(max_sample_x):
        for __j_y in range(max_sample_y):   
            alpha_sample_zero = alpha[__i_x]
            beta_sample_zero = beta[__j_y]
            # commet: (x_0-x_1)^2 + (y_0-y_1)^2 + (z_0 - z_1)^2
            l2_x = (alpha_sample_zero[:, 0] - beta_sample_zero[:, 0]) ** 2
            l2_y = (alpha_sample_zero[:, 1] - beta_sample_zero[:, 1]) ** 2
            l2_z = (alpha_sample_zero[:, 2] - beta_sample_zero[:, 2]) ** 2            
            l2_total = (l2_x + l2_y + l2_z).sum()
            
            assert (d_container.d_xy[__i_x][__j_y] - l2_total) < 1e-5
        # end for
    # end for