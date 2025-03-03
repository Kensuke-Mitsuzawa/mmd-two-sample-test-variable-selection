import typing as ty
import random
import logging

from pathlib import Path

import torch
import numpy as np
from statsmodels.tsa.stattools import adfuller

from ...logger_unit import handler

logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


"""This module is for finding proper time-slicing points for a time-series.
A time-slicing point is supposed to pass Augmented Dickey-Fuller unit root test (ADF test).
"""


def __split_time_bucket(
    array_x: np.ndarray,
    array_y: np.ndarray,
    t_current_start_time_bucket: int,
    n_span_time_slicing: int,
    test_aggregation_mode: ty.Optional[str] = None,
    ratio_sampling_sensor: float = 0.5,
    threshold_ratio_sensors: float = 0.8,
    n_time_slicing_tuning: int = 10) -> int:
    """I find a time-step to taking out a specific time-bucket.
    """
    seq_sensor_id = range(0, array_x.shape[0])

    is_stationary_time_bucket: bool = False
    
    while True:
        # slicing time-steps for a time-bucket.
        array_time_bucket_x = array_x[:, t_current_start_time_bucket:(t_current_start_time_bucket + n_span_time_slicing)]
        array_time_bucket_y = array_y[:, t_current_start_time_bucket:(t_current_start_time_bucket + n_span_time_slicing)]

        if test_aggregation_mode is None:
            _seq_sensor_id_sample = random.sample(seq_sensor_id, k=int(len(seq_sensor_id) * ratio_sampling_sensor))

            _seq_test_binary_flags = []
            for _sensor_id in _seq_sensor_id_sample:
                _array_sensor_value_x = array_time_bucket_x[_sensor_id, :]
                __res_tuples_x = adfuller(_array_sensor_value_x)
                _pvalue_x = __res_tuples_x[1]

                _array_sensor_value_y = array_time_bucket_y[_sensor_id, :]
                __res_tuples_y = adfuller(_array_sensor_value_y)
                _pvalue_y = __res_tuples_y[1]                
                
                # Null-Hypothesis or not? H0 of ADF is non-stationary sequence. Hence, reject H0 means stationary.  
                _is_null_h0_x = True if _pvalue_x <= 0.05 else False
                _is_null_h0_y = True if _pvalue_y <= 0.05 else False

                _is_stationary_bucket = True if _is_null_h0_x and _is_null_h0_y else False
                _seq_test_binary_flags.append(_is_stationary_bucket)
            # end for

            _ratio_sensors_test_passing: float = len([__flag for __flag in _seq_test_binary_flags if __flag is True]) / len(_seq_test_binary_flags)
            is_stationary_time_bucket = True if _ratio_sensors_test_passing > threshold_ratio_sensors else False
        else:
            _array_sum_time_bucket_x = np.sum(array_time_bucket_x, axis=0)
            __res_tuples_x = adfuller(_array_sum_time_bucket_x)

            _array_sum_time_bucket_y = np.sum(array_time_bucket_y, axis=0)
            __res_tuples_y = adfuller(_array_sum_time_bucket_y)
            
            _pvalue_x = __res_tuples_x[1]
            _pvalue_y = __res_tuples_y[1]
                
            # Null-Hypothesis or not?
            _is_null_h0_x = True if _pvalue_x <= 0.05 else False
            _is_null_h0_y = True if _pvalue_y <= 0.05 else False            

            is_stationary_time_bucket = True if _is_null_h0_x and _is_null_h0_y else False
        # end if

        if is_stationary_time_bucket is True:
            break
        # end if

        # updating parameter for the next iteration.
        # updating the length of a time-bucket.
        n_span_time_slicing = n_span_time_slicing - n_time_slicing_tuning
        
        if n_span_time_slicing < 0:
            logger.warning(f'Cannot find a proper time-slicing point. n_span_time_slicing is negative.')
            n_span_time_slicing = 1
            break
        # end if
    # end while

    return t_current_start_time_bucket + n_span_time_slicing
# end def


def main(
    array_x: ty.Union[np.ndarray, Path],
    array_y: ty.Union[np.ndarray, Path],
    n_span_time_slicing: int = 100,
    test_aggregation_mode: ty.Optional[str] = None,
    ratio_sampling_sensor: float = 0.5,
    threshold_ratio_sensors: float = 0.8,
    n_time_slicing_tuning: int = 10) -> ty.List[int]:
    """
    Returns
    ----------
    a list of time-step where a time-bucket ends.
    """
    # loading array_x and array_y.
    if isinstance(array_x, Path):
        if array_x.suffix == '.pt':
            __dict_x = torch.load(array_x)
        elif array_x.suffix == '.np':
            __dict_x = np.load(array_x)
        else:
            raise ValueError(f'array_x must be a .pt or .np file.')
        array_x = __dict_x['array']
    else:
        pass
    # end if

    if isinstance(array_y, Path):
        if array_y.suffix == '.pt':
            __dict_y = torch.load(array_y)
        elif array_y.suffix == '.np':
            __dict_y = np.load(array_y)
        else:
            raise ValueError(f'array_y must be a .pt or .np file.')
        array_y = __dict_y['array']
    else:
        pass
    # end if

    # a list to save time-steps of a time-bucket.
    seq_time_bucket_split = []

    n_timesteps = array_x.shape[1]
    assert n_timesteps == array_y.shape[1], f'array_x and array_y must have the same timesteps'

    n_current_time_slicing = 0

    while n_current_time_slicing < n_timesteps:
        __t_end_time_bucket = __split_time_bucket(
            array_x=array_x,
            array_y=array_y,
            t_current_start_time_bucket=n_current_time_slicing,
            n_span_time_slicing=n_span_time_slicing,
            test_aggregation_mode=test_aggregation_mode,
            ratio_sampling_sensor=ratio_sampling_sensor,
            threshold_ratio_sensors=threshold_ratio_sensors,
            n_time_slicing_tuning=n_time_slicing_tuning)
        
        seq_time_bucket_split.append(__t_end_time_bucket)
        n_current_time_slicing = n_current_time_slicing + __t_end_time_bucket
    # end while

    return seq_time_bucket_split