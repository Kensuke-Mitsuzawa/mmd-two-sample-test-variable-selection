import typing
import logging

import torch
import torch.nn
import numpy as np

from distributed import Client

import optuna

from ..datasets.base import BaseDataset

from ..distance_module.base import BaseDistanceModule
from ..distance_module.time_aware_distance import TimeAwareDistance, DistanceContainer
from .base import (BaseKernel, KernelMatrixObject)
from .commons import (QuadraticKernelMatrixContainer, LinearKernelMatrixContainer)
from . import utils
from ..logger_unit import handler

logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


class DistributedFunctionArg(typing.NamedTuple):
    index_dimension: int
    n_dimension: int 
    x_projected_m: torch.tensor 
    y_projected_m: torch.tensor
    heuristic_operation: str
    distance_function: typing.Callable[[torch.Tensor, torch.Tensor], DistanceContainer]
    
    
    
def compute_length_scale_dataset_dimension_d(args: DistributedFunctionArg) -> typing.Tuple[int, torch.Tensor]:
    """Ptivate API. Compute a length scale for a dimension.
    
    Parameters
    ----------
    args: DistributedFunctionArg
        
    Returns
    -------
    A tuple (int, torch.Tensor).
    
    torch.Tensor
        A length scale for a dimension.
    """
    x_projected_m = args.x_projected_m
    y_projected_m = args.y_projected_m
    n_dimension = args.n_dimension

    d_function = args.distance_function
    heuristic_operation = args.heuristic_operation

    __d_container = d_function(x_projected_m, y_projected_m)
    a_m = __d_container.d_xx
    b_m = __d_container.d_yy
    c_m = __d_container.d_xy
    
    if heuristic_operation == 'median':
        gamma_m: torch.Tensor = torch.median(torch.cat([a_m, b_m, c_m]))
    elif heuristic_operation == 'mean':
        gamma_m: torch.Tensor = torch.mean(torch.cat([a_m, b_m, c_m]))
    else:
        raise Exception(f'heuristic_operation == {heuristic_operation} does not exist.')
    # end if

    return args.index_dimension, gamma_m * n_dimension



class TimeAwareQuadraticKernelGaussianKernel(BaseKernel):
    def __init__(self,
                 distance_module: BaseDistanceModule = TimeAwareDistance(coordinate_size=1),
                 transition_graph_weights: typing.Optional[torch.Tensor] = None,
                 generation_weights: typing.Optional[torch.Tensor] = None,
                 constraints_graph_weights: typing.Optional[torch.Tensor] = None,
                 dask_client: typing.Optional[Client] = None,
                 bandwidth: typing.Optional[torch.Tensor] = None,
                 n_timestamp: typing.Optional[int] = None,
                 n_sensors: typing.Optional[int] = None):
        """
        Parameters
        ----------
        distance_module: `TimeAwareDistance`
        transition_graph_weights: `torch.Tensor`
            The tensor size is (|T|-1, |S|, |S|)
        generation_weights: `torch.Tensor`
            The tensor size is (|S|, |T|)
        constraints_graph_weights: `torch.Tensor`
            The tensor size is (|S|, |S|).
            The adjacency matrix of the constraints graph.
            Values are either 1.0 and 0.0 or weights of edges.
        """
        assert isinstance(distance_module, TimeAwareDistance), f'Distance module must be `TimeAwareDistance`.'
        
        # ------------------------------
        # attributes
        self.transition_graph_weights: typing.Optional[torch.Tensor] = None
        self.generation_weights: typing.Optional[torch.Tensor] = None
        self.constraints_graph_weights: typing.Optional[torch.Tensor] = None
        
        self.shape_transition_graph_weights: typing.Optional[typing.Tuple[int, int, int]] = None
        self.shape_generation_weights: typing.Optional[typing.Tuple[int, int]] = None
        
        self.n_timestamp = n_timestamp
        self.n_sensors = n_sensors
        # ------------------------------
        
        super().__init__(
            distance_module=distance_module,
            possible_shapes=(2, 3),
            bandwidth=bandwidth,
            kernel_computation_type='quadratic',
            ard_weights=torch.ones(1))
        self.dask_client = dask_client
        
        self.transition_graph_weights = transition_graph_weights
        self.generation_weights = generation_weights
        self.constraints_graph_weights = constraints_graph_weights
        
        self.__init_weights()
        
    def __init_weights(self):
        """
        """
        if self.transition_graph_weights is None and self.generation_weights is None:
            assert self.n_sensors is not None and self.n_timestamp is not None, \
                'n_sensors, n_timestamp must be given when transition_graph_weights and generation_weights are None.'
        # end if
        
        # comment: initializing all 1.0
        if self.constraints_graph_weights is None:
            self.constraints_graph_weights = torch.ones((self.n_sensors, self.n_sensors))
        # end if
        
        if self.transition_graph_weights is None:
            self.transition_graph_weights = torch.nn.Parameter(torch.ones((self.n_timestamp - 1, self.n_sensors, self.n_sensors)))
        else:
            self.transition_graph_weights = torch.nn.Parameter(self.transition_graph_weights)
        # end if
        
        if self.generation_weights is None:
            self.generation_weights = torch.nn.Parameter(torch.ones((self.n_sensors, self.n_timestamp)))
        else:
            self.generation_weights = torch.nn.Parameter(self.generation_weights)
        # end if

    @classmethod
    def from_dataset(cls, 
                     dataset: BaseDataset,
                     distance_module: BaseDistanceModule = TimeAwareDistance(coordinate_size=1),
                     bandwidth: typing.Optional[torch.Tensor] = None,
                     ard_weights: typing.Optional[torch.Tensor] = None,
                     heuristic_operation: str = 'median',
                     is_dimension_median_heuristic: bool = True,
                     dask_client: typing.Optional[Client] = None
                     ) -> "TimeAwareQuadraticKernelGaussianKernel":
        """Public API. Create a kernel object from a dataset.
        """
        # do kernel length initialization.
        if ard_weights is None:
            __dim_x, __dim_y = dataset.get_dimension_flattened()
            ard_weights = torch.ones(__dim_x)
        # end if
        
        kernel_obj = cls(
            distance_module=distance_module,
            bandwidth=bandwidth,
            ard_weights=ard_weights,
            heuristic_operation=heuristic_operation,
            is_dimension_median_heuristic=is_dimension_median_heuristic,
            opt_bandwidth=False,
            dask_client=dask_client)
        
        # do kernel length initialization.
        kernel_obj.compute_length_scale_dataset(dataset)
        
        return kernel_obj
        
    def get_hyperparameters(self) -> typing.Dict[str, typing.Any]:
        raise NotImplementedError('need to be implemented.')
        return {}

    def _get_trainable_parameters(self) -> typing.List[torch.nn.Parameter]:
        return [self.transition_graph_weights, self.generation_weights]

    def _get_median_single(self,
                           x: torch.Tensor,
                           y: torch.Tensor) -> torch.Tensor:
        weights = self.__get_compute_weights()
        # case: x and y may be flattened. I reconstruct 2D data array from 1D array.
        if len(x.shape) == 2 and len(y.shape) == 2:
            x = torch.reshape(x, (x.shape[0], self.n_sensors, self.n_timestamp))
            y = torch.reshape(y, (y.shape[0], self.n_sensors, self.n_timestamp))
        else:
            x = x
            y = y
        # end if
        
        # comment: newly introduced implementation. using a module.
        x_ard = torch.mul(weights, x)
        y_ard = torch.mul(weights, y)
        
        d_container = self.distance_module.compute_distance(x_ard, y_ard)

        d_xx_ard_gamma = d_container.d_xx
        d_yy_ard_gamma = d_container.d_yy
        d_xy_ard_gamma = d_container.d_xy
        
        if self.heuristic_operation == 'median':
            value_length_scale = torch.median(torch.stack([
                d_xx_ard_gamma.flatten(),
                d_yy_ard_gamma.flatten(),
                d_xy_ard_gamma.flatten()
            ]))
        elif self.heuristic_operation == 'mean':
            value_length_scale = torch.mean(torch.stack([
                d_xx_ard_gamma.flatten(),
                d_yy_ard_gamma.flatten(),
                d_xy_ard_gamma.flatten()
            ]))
        else:
            raise Exception(f'is_dimension_median_heuristic == {self.heuristic_operation} does not exist.')
        # end if
        
        return value_length_scale
        

    def _get_median_dim(self,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        is_completion_missing: bool = True
                        ) -> typing.Optional[torch.Tensor]:
        """
        """
        weights = self.__get_compute_weights()
        
        n_timestamp = weights.shape[1]
        n_sensors = weights.shape[0]

        # case: x and y may be flattened. I reconstruct 2D data array from 1D array.
        if len(x.shape) == 2 and len(y.shape) == 2:
            x = torch.reshape(x, (x.shape[0], self.n_sensors, self.n_timestamp))
            y = torch.reshape(y, (y.shape[0], self.n_sensors, self.n_timestamp))
        else:
            x = x
            y = y
        # end if

        x_ard = torch.mul(weights, x)
        y_ard = torch.mul(weights, y)
        
    
        tensor_length_scale = torch.zeros((n_sensors, n_timestamp))
                
        # length-scale computing per timestep.
        # ------------------------------
        for __i_sensor in range(n_sensors):
            for __i_time_step in range(n_timestamp):
                __timesteps = list(range(0, __i_time_step + 1))
                __x_at_t = x_ard[:, __i_sensor, __timesteps]
                __y_at_t = y_ard[:, __i_sensor, __timesteps]
                
                # comment: 2d tensor -> 3d tensor
                x_3d = __x_at_t[:, None, :]
                y_3d = __y_at_t[:, None, :]
            
                d_container = self.distance_module.compute_distance(x_3d, y_3d)

                d_xx_ard_gamma = d_container.d_xx
                d_yy_ard_gamma = d_container.d_yy
                d_xy_ard_gamma = d_container.d_xy
            
                if self.heuristic_operation == 'median':
                    value_length_scale = torch.median(torch.stack([
                        d_xx_ard_gamma.flatten(),
                        d_yy_ard_gamma.flatten(),
                        d_xy_ard_gamma.flatten()
                    ]))
                elif self.heuristic_operation == 'mean':
                    value_length_scale = torch.mean(torch.stack([
                        d_xx_ard_gamma.flatten(),
                        d_yy_ard_gamma.flatten(),
                        d_xy_ard_gamma.flatten()
                    ]))
                else:
                    raise Exception(f'is_dimension_median_heuristic == {self.heuristic_operation} does not exist.')
                # end if
                
                tensor_length_scale[__i_sensor, __i_time_step] = value_length_scale
            # end for
        # end for
        
        # ------------------------------
        # tuning length-scale.
        tensor_length_scale = self.__tune_length_scale_dimension_wise(
            x=x,
            y=y,
            weights=weights,
            length_scale=tensor_length_scale)
        # ------------------------------
        
        return tensor_length_scale
    
    def __tune_length_scale_dimension_wise(self,
                                           x: torch.Tensor,
                                           y: torch.Tensor,
                                           weights: torch.Tensor,
                                           length_scale: torch.Tensor) -> torch.Tensor:
        """Private API. Tune length scale dimension-wise.
        The calculated length scale may not be appropriate.
        So, I compute a length scale that Kernel-matrix has a value within a range.
        """    
        x_ard = torch.mul(weights, x)
        y_ard = torch.mul(weights, y)
        
        length_scale_tuned: torch.Tensor
        value_tuning = torch.arange(1.0, 100.0, 1.0)
        for __v_tuning in value_tuning:
            length_scale_tuned = length_scale * __v_tuning
            
            __x_ard_gamma = torch.div(x_ard, torch.sqrt(length_scale_tuned))
            __y_ard_gamma = torch.div(y_ard, torch.sqrt(length_scale_tuned))

            __d_container = self.distance_module.compute_distance(__x_ard_gamma, __y_ard_gamma)        
            k_xx = torch.exp(-1 * __d_container.d_xx)
            k_yy = torch.exp(-1 * __d_container.d_yy)
            __k_xy = torch.exp(-1 * __d_container.d_xy)
            
            stats_k_xy = torch.median(__k_xy)
            if stats_k_xy > 0.01 and stats_k_xy < 1.0:
                break
            # end if
        # end for
        
        return length_scale_tuned

    def _compute_kernel_matrix_single(self,
                                      x: torch.Tensor,
                                      y: torch.Tensor,
                                      bandwidth: typing.Optional[torch.Tensor] = None
                                      ) -> KernelMatrixObject:
        raise NotImplementedError('This method is not implemented.')
    
    def __get_compute_weights(self) -> torch.Tensor:
        # a part of forwarding step.
        n_sensors = self.generation_weights.shape[0]
        n_timestamp = self.generation_weights.shape[1]
        
        weights = torch.zeros((n_sensors, n_timestamp))
        # computing weights for every timestep.
        for __i_t in range(n_timestamp):
            if __i_t == 0:
                weights[:, __i_t] = self.generation_weights[:, __i_t]
            else:
                # product of transition weights and generation weights
                __index_transition = __i_t - 1
                # tensor of (|S|, |S|)
                __weight_transition_at_t = self.transition_graph_weights[__index_transition]
                # masking constraints. tensor of (|S|, |S|)
                __weights_transition_masked = torch.mul(self.constraints_graph_weights, __weight_transition_at_t)
                # (|S|, |S|) \dot (|S|) -> (|S|)
                weights_at_t = torch.matmul(__weights_transition_masked, self.generation_weights[:, __i_t])
                weights[:, __i_t] = weights_at_t
            # end if
        # end for
        
        return weights

    def _compute_kernel_matrix_dim(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   bandwidth: typing.Optional[torch.Tensor] = None) -> KernelMatrixObject:
        if bandwidth is None:
            bandwidth = self.bandwidth
        # end if
        assert bandwidth is not None, 'bandwidth is None. Use set_length_scale(training_dataset) to set the bandwidth.'
        
        # ------------------------------
        # computing weights.
        weights = self.__get_compute_weights()
        # ------------------------------
        # case: x and y may be flattened. I reconstruct 2D data array from 1D array.
        if len(x.shape) == 2 and len(y.shape) == 2:
            x = torch.reshape(x, (x.shape[0], self.n_sensors, self.n_timestamp))
            y = torch.reshape(y, (y.shape[0], self.n_sensors, self.n_timestamp))
        else:
            x = x
            y = y
        # end if
        
        # comment: newly introduced implementation. using a module.
        x_ard = torch.mul(weights, x)
        y_ard = torch.mul(weights, y)
        
        x_ard_gamma = torch.div(x_ard, torch.sqrt(bandwidth))
        y_ard_gamma = torch.div(y_ard, torch.sqrt(bandwidth))

        d_container = self.distance_module.compute_distance(x_ard_gamma, y_ard_gamma)

        d_xx_ard_gamma = d_container.d_xx
        d_yy_ard_gamma = d_container.d_yy
        d_xy_ard_gamma = d_container.d_xy

        k_xx = torch.exp(-1 * d_xx_ard_gamma)
        k_yy = torch.exp(-1 * d_yy_ard_gamma)
        k_xy = torch.exp(-1 * d_xy_ard_gamma)
        
        k_container = QuadraticKernelMatrixContainer(k_xx, k_yy, k_xy)
        return KernelMatrixObject(kernel_computation_type=self.kernel_computation_type, x_size=len(x), y_size=len(y),
                                  kernel_matrix_container=k_container)
