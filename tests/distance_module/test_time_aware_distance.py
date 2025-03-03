import typing as ty
import numpy as np
import torch

from mmd_tst_variable_detector.distance_module.time_aware_distance import TimeAwareDistance


def __generate_time_aware_distance(n_sample: int = 5,
                                   n_timstamp: int = 5,
                                   n_sensor: int = 10):
    """Generate a TimeAwareDistance object. Pseudo data is generated."""
    set_x = torch.zeros((n_sample, n_sensor, n_timstamp))
    set_y = torch.zeros((n_sample, n_sensor, n_timstamp))
    
    for __i_sample in range(n_sample):
        x = torch.randn(n_sensor, n_timstamp)
        y = torch.randn(n_sensor, n_timstamp)
        
        set_x[__i_sample] = x
        set_y[__i_sample] = y
    # end for
    return set_x, set_y
    
    
def test_time_aware_distance():
    """Testing time-aware distance function."""
    set_x, set_y = __generate_time_aware_distance()
    func_distance = TimeAwareDistance()
    distance_container = func_distance.compute_distance(set_x, set_y)
    
    assert distance_container.d_xx.shape == (5, 5)
    assert distance_container.d_xy.shape == (5, 5)
    assert distance_container.d_yy.shape == (5, 5)
