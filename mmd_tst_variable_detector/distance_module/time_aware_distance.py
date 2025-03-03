import typing

import numpy as np
import torch
import torch.jit

from .base import BaseDistanceModule, DistanceContainer


# @torch.jit.script  # type: ignore
def distance_x_y(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computing distance between x and y, paying attention to the time aspect dimension.
    
    The formulation is $$Distance(x, y) = \sum_{t=0}^{T} \sum_{s=0}^{S} (x_{t, s} - y_{t, s})^2$$.
    
    Parameters
    ----------
    x: (|S|, |T|)
    y: (|S|, |T|)
    
    Return
    ----------
    A 0-d tensor. A distance value.
    """
    n_time_stamps = x.shape[1]
    n_sensors = x.shape[0]
    
    total_d_value = torch.tensor(0.0)
    # TODO rewriting `torch.pdist(x_reshaped)`. May be faster than double-for-looping.
    for __id_time_stamp in range(0, n_time_stamps):
        __total_sensors = torch.tensor(0.0)
        for __id_sensor in range(0, n_sensors):
            __x_value = x[__id_sensor, __id_time_stamp]
            __y_value = y[__id_sensor, __id_time_stamp]
            __d_value = (__x_value - __y_value) ** 2
            __total_sensors = __total_sensors + __d_value
        # end for
        total_d_value += __total_sensors
    # end for
    
    return total_d_value    



class TimeAwareDistance(BaseDistanceModule):
    """Time-aware distance function."""
    def __init__(self,
                 coordinate_size: int = 1) -> None:
        if coordinate_size == 1:    
           data_point_shape = (1,)
        else:
            raise ValueError("coordinate_size must be 1.")
        # end if
        
        super().__init__(data_point_shape)
    
    def __compute_d_matrix_same(self, x: torch.Tensor) -> torch.Tensor:
        """Distance matrix only when x and y have the same number of data.
        Less computation thanks to a triangular matrix.
        
        Parameters
        ----------
        x: (N, |S|, |T|)
        
        Return
        ----------
        A 2nd tensor (N, N)
        """
        x_samples = len(x)
        
        tri_torch = torch.zeros((x_samples, x_samples))
        target_indices = torch.triu_indices(tri_torch.shape[0], tri_torch.shape[1], offset=1)
        for __index_triangular in zip(target_indices[0], target_indices[1]):
            __ind_row = __index_triangular[0]
            __ind_col = __index_triangular[1]
            __d_value = distance_x_y(x[__ind_row], x[__ind_col])
            tri_torch[__ind_row, __ind_col] = __d_value
        # end for

        index_lower = torch.tril_indices(tri_torch.shape[0], tri_torch.shape[1])
        tri_torch[index_lower[0], index_lower[1]] = tri_torch.T[index_lower[0], index_lower[1]]

        return tri_torch

    def __compute_d_matrix_xy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Distance matrix between x and y.
        
        Parameters
        ----------
        x: (N, |S|, |T|)
        y: (M, |S|, |T|)
        
        Return
        ----------
        A 2nd tensor (N, M)
        """
        x_samples = len(x)
        y_samples = len(y)
        
        d_xy = torch.zeros((x_samples, y_samples))
        
        for __i_x in range(0, x_samples):
            for __i_y in range(0, y_samples):                
                __d_value = distance_x_y(x[__i_x], y[__i_y])
                d_xy[__i_x, __i_y] = __d_value
            # end for
        # end for        
        
        return d_xy

    def __compute_l2_flat_vector_alpha(self, x: torch.Tensor, y: torch.Tensor) -> DistanceContainer:
        """Private API. Computing Distance Matrix by a time-aware distance function."""
        d_xx_sqrt = self.__compute_d_matrix_same(x)
        d_yy_sqrt = self.__compute_d_matrix_same(y)
        d_xy_sqrt = self.__compute_d_matrix_xy(x, y)
                
        return DistanceContainer(d_xx_sqrt, d_yy_sqrt, d_xy_sqrt)        
            
    # ----------------------------------------------------------------------------
    # Public API
        
    def get_hyperparameters(self) -> typing.Dict[str, typing.Any]:
        return {
            "coordinate_size": self.data_point_shape[0]
        }
        
    def compute_distance(self, x: torch.Tensor, y: torch.Tensor) -> DistanceContainer:
        """Public API. Compute distance between x and y.
        
        X and Y must be samples from different probabilistic distributions.
        In other words, X ~ P, Y ~ Q.
        
        Args:
            x: (N, |S|, |T|)
            y: (N, |S|, |T|)
        
        Returns:
            A `DistanceContainer` container. The container contains d_xx, d_yy, and d_xy. All d_** are in (N, N).
        """
        assert len(x.shape) == 3, "x must be a 3D tensor. But the given shape is {}".format(x.shape)
        assert len(y.shape) == 3, "y must be a 3D tensor. But the given shape is {}".format(y.shape)
        
        if self.data_point_shape == (1,):
            # comment: implementation alpha is faster than beta.
            d_container_alpha = self.__compute_l2_flat_vector_alpha(x, y)
            return d_container_alpha
        else:
            raise ValueError("data_point_shape must be 1")
