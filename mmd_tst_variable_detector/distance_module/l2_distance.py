import typing

import numpy as np
import torch

from .base import BaseDistanceModule, DistanceContainer
from ..kernels.utils import distance_over_3rd_reshape_same_data, distance_over_3rd_reshape_xy_data



class L2Distance(BaseDistanceModule):
    def __init__(self,
                 coordinate_size: int) -> None:
        if coordinate_size == 1:    
           data_point_shape = (1,)
        elif coordinate_size == 2:
            data_point_shape = (2,)
        elif coordinate_size == 3:
            data_point_shape = (3,)
        else:
            raise ValueError("coordinate_size must be 1, 2, or 3.")
        # end if
        self.coordinate_size = coordinate_size
        
        super().__init__(data_point_shape)
    
    @staticmethod
    @torch.jit.script  # type: ignore
    def __compute_l2_flat_vector_alpha(x: torch.Tensor, y: torch.Tensor) -> DistanceContainer:
        """Compute L2 distance between two flat vectors.
        
        Args:
            x: (N, 1)
            y: (N, 1)
            
        Returns:
            `DistanceContainer(d_xx, d_yy, d_xy)`. d_** is (N, N).
        """
        # torch.t() is transpose function. torch.dot() is only for vectors. For 2nd tensors, "mm".
        xx = torch.mm(x, torch.t(x))
        xy = torch.mm(x, torch.t(y))
        yy = torch.mm(y, torch.t(y))

        x_sqnorms = torch.diagonal(xx, offset=0)
        y_sqnorms = torch.diagonal(yy, offset=0)

        d_xy_sqrt = (-2 * xy + x_sqnorms[:, np.newaxis] + y_sqnorms[np.newaxis, :])
        d_xx_sqrt = (-2 * xx + x_sqnorms[:, np.newaxis] + x_sqnorms[np.newaxis, :])
        d_yy_sqrt = (-2 * yy + y_sqnorms[:, np.newaxis] + y_sqnorms[np.newaxis, :])
        
        d_xy = d_xy_sqrt
        d_xx = d_xx_sqrt
        d_yy = d_yy_sqrt
        
        return DistanceContainer(d_xx, d_yy, d_xy)

    def __compute_l2_flat_vector_beta(self, x: torch.Tensor, y: torch.Tensor) -> DistanceContainer:
        # Note: d_**_ard_gamma is L2-norm. Therefore, it should be square.
        d_xx_sqrt = distance_over_3rd_reshape_same_data(x)
        d_yy_sqrt = distance_over_3rd_reshape_same_data(y)
        d_xy_sqrt = distance_over_3rd_reshape_xy_data(x, y)
        
        d_xx = d_xx_sqrt ** 2
        d_yy = d_yy_sqrt ** 2
        d_xy = d_xy_sqrt ** 2
        
        return DistanceContainer(d_xx, d_yy, d_xy)
    
    def __compute_l2_xy_coordinates(self, x: torch.Tensor, y: torch.Tensor, is_compute_length_scale: bool) -> DistanceContainer:
        """
        A data feature consists of (|A|, 2), A is the set of agents. A part of data features. 
        
        Args:
            x: (N, |A|, C), C=2
            y: (N, |A|, C), C=2        
        """
        # xy coordinate from the variable x
        if is_compute_length_scale:
            # comment: length scale computation has a shape of (n-sample, n-coordinate)
            x_position_0 = x[:, None, 0]
            x_position_1 = x[:, None, 1]
        else:
            x_position_0 = x[:, :, 0]
            x_position_1 = x[:, :, 1]
        # end if

        # xy coordinate from the variable y
        if is_compute_length_scale:
            y_position_0 = y[:, None, 0]
            y_position_1 = y[:, None, 1]        
        else:
            y_position_0 = y[:, :, 0]
            y_position_1 = y[:, :, 1]
        # end if

        # L2 distance on the x coordinate
        d_container_position_zero = self.__compute_l2_flat_vector_alpha(x_position_0, y_position_0)
        # L2 distance on the y coordinate
        d_container_position_one = self.__compute_l2_flat_vector_alpha(x_position_1, y_position_1)
        # end if        
        d_xx = d_container_position_zero.d_xx + d_container_position_one.d_xx
        d_yy = d_container_position_zero.d_yy + d_container_position_one.d_yy
        d_xy = d_container_position_zero.d_xy + d_container_position_one.d_xy

        return DistanceContainer(d_xx, d_yy, d_xy)

    def __compute_l2_xyz_coordinates(self, x: torch.Tensor, y: torch.Tensor, is_compute_length_scale: bool) -> DistanceContainer:
        """
        Args:
            x: (N, |A|, C), C=3
            y: (N, |A|, C), C=3
        """
        if is_compute_length_scale:
            # xy coordinate from the variable x
            x_position_0 = x[:, None, 0]
            x_position_1 = x[:, None, 1]
            x_position_2 = x[:, None, 2]
        else:
            # xy coordinate from the variable x
            x_position_0 = x[:, :, 0]
            x_position_1 = x[:, :, 1]
            x_position_2 = x[:, :, 2]        
            
        if is_compute_length_scale:
            # xy coordinate from the variable y
            y_position_0 = y[:, None, 0]
            y_position_1 = y[:, None, 1]
            y_position_2 = y[:, None, 2]
        else:
            # xy coordinate from the variable y
            y_position_0 = y[:, :, 0]
            y_position_1 = y[:, :, 1]
            y_position_2 = y[:, :, 2]
        # end if

        # L2 distance on the x coordinate
        d_container_position_0 = self.__compute_l2_flat_vector_alpha(x_position_0, y_position_0)
        # L2 distance on the y coordinate
        d_container_position_1 = self.__compute_l2_flat_vector_alpha(x_position_1, y_position_1)
        # L2 distance on the z coordinate
        d_container_position_2 = self.__compute_l2_flat_vector_alpha(x_position_2, y_position_2)
        
        d_xx = d_container_position_0.d_xx + d_container_position_1.d_xx + d_container_position_2.d_xx
        d_yy = d_container_position_0.d_yy + d_container_position_1.d_yy + d_container_position_2.d_yy
        d_xy = d_container_position_0.d_xy + d_container_position_1.d_xy + d_container_position_2.d_xy

        return DistanceContainer(d_xx, d_yy, d_xy)
        
    # ----------------------------------------------------------------------------
    # Public API
        
    def get_hyperparameters(self) -> typing.Dict[str, typing.Any]:
        return {
            "coordinate_size": self.data_point_shape[0]
        }
        
    def compute_distance(self, 
                         x: torch.Tensor, 
                         y: torch.Tensor, 
                         is_compute_length_scale: bool = False) -> DistanceContainer:
        """Public API. Compute distance between x and y.
        
        X and Y must be samples from different probabilistic distributions.
        In other words, X ~ P, Y ~ Q.
        
        Args:
            x: (N, |A|) or (N, |A|, C), C=2, 3
            y: (N, |A|) or (N, |A|, C), C=2, 3
        
        Returns:
            A `DistanceContainer` container. The container contains d_xx, d_yy, and d_xy. All d_** are in (N, N).
        """
        if self.data_point_shape == (1,):
            # comment: implementation alpha is faster than beta.
            d_container_alpha = self.__compute_l2_flat_vector_alpha(x, y)
            return d_container_alpha
        elif self.data_point_shape == (2,):
            d_container_alpha = self.__compute_l2_xy_coordinates(x, y, is_compute_length_scale)
            return d_container_alpha
        elif self.data_point_shape == (3,):
            d_container_alpha = self.__compute_l2_xyz_coordinates(x, y, is_compute_length_scale)
            return d_container_alpha
        else:
            raise ValueError("data_point_shape must be (1,), (2,), or (3,).")
