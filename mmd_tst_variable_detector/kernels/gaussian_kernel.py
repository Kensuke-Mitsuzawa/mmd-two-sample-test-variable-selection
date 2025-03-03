import typing
import logging

import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from distributed import Client

from ..datasets.base import BaseDataset

from ..distance_module.base import BaseDistanceModule
from ..distance_module.l2_distance import L2Distance, DistanceContainer
from .base import (BaseKernel, KernelMatrixObject)
from .commons import (QuadraticKernelMatrixContainer, LinearKernelMatrixContainer)
from . import utils
from .. import logger_unit

logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(logger_unit.handler)


class DistributedFunctionArg(typing.NamedTuple):
    index_dimension: int
    n_dimension: int 
    x_projected_m: torch.Tensor 
    y_projected_m: torch.Tensor
    heuristic_operation: str
    distance_function: typing.Callable[[torch.Tensor, torch.Tensor, bool], DistanceContainer]
    
    
    
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

    __d_container = d_function(x_projected_m, y_projected_m, True)
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



class QuadraticKernelGaussianKernel(BaseKernel):
    def __init__(self,
                 distance_module: BaseDistanceModule = L2Distance(coordinate_size=1),
                 bandwidth: typing.Optional[torch.Tensor] = None,
                 ard_weights: typing.Optional[torch.Tensor] = None,
                 ard_weight_shape: typing.Optional[typing.Tuple[int, ...]] = None,
                 is_force_cutoff: bool = False,
                 ratio_cutoff: float = -1,
                 heuristic_operation: str = 'median',
                 is_auto_adjust_gamma: bool = False,
                 is_dimension_median_heuristic: bool = True,
                 opt_bandwidth: bool = False,
                 dask_client: typing.Optional[Client] = None):
        """
        Parameters
        ----------
        distance_module: BaseDistanceModule
            distance module object.
        bandwidth: torch.Tensor
            bandwidth for Gaussian kernel.
        ard_weights: torch.Tensor
            ARD weights for Gaussian kernel.
        ard_weight_shape: typing.Optional[typing.Tuple[int, ...]]
            ARD weight shape. If None, the shape is automatically determined.
        is_force_cutoff: bool
            If True, the kernel matrix is forced to be positive definite.
        ratio_cutoff: float
            If is_force_cutoff is True, the kernel matrix is forced to be positive definite.
            The ratio_cutoff is a ratio of the minimum eigenvalue to the maximum eigenvalue.
            If the ratio is smaller than the ratio_cutoff, the kernel matrix is forced to be positive definite.
        heuristic_operation: str
            'median' or 'mean'. The heuristic operation for median heuristic.
        is_auto_adjust_gamma: bool
            If True, the gamma is automatically adjusted to realize acceptable K_xy values.
        is_dimension_median_heuristic: bool
            If True, the median heuristic is computed for each dimension.
        opt_bandwidth: bool
            If True, the bandwidth is optimized.
        dask_client: typing.Optional[Client]
            Dask client object. Used for computing the initial length scale (bandwidth).
        """
        super().__init__(
            distance_module=distance_module,
            possible_shapes=(2, 3),
            bandwidth=bandwidth,
            ard_weights=ard_weights,
            ard_weight_shape=ard_weight_shape,
            is_force_cutoff=is_force_cutoff,
            ratio_cutoff=ratio_cutoff,
            heuristic_operation=heuristic_operation,
            is_auto_adjust_gamma=is_auto_adjust_gamma,
            is_dimension_median_heuristic=is_dimension_median_heuristic,
            opt_bandwidth=opt_bandwidth,
            kernel_computation_type='quadratic'
        )
        self.dask_client = dask_client
        
    @classmethod
    def from_dataset(cls, 
                     dataset: BaseDataset,
                     distance_module: BaseDistanceModule = L2Distance(coordinate_size=1),
                     bandwidth: typing.Optional[torch.Tensor] = None,
                     ard_weights: typing.Optional[torch.Tensor] = None,
                     heuristic_operation: str = 'median',
                     is_dimension_median_heuristic: bool = True,
                     dask_client: typing.Optional[Client] = None
                     ) -> "QuadraticKernelGaussianKernel":
        """Public API. Create a kernel object from a dataset.
        """
        # do kernel length initialization.
        if ard_weights is None:
            _t_data_dims = dataset.get_dimension_data_space()
            ard_weights = torch.ones(_t_data_dims)
        # end if
        
        assert ard_weights is not None, 'ard_weights is None.'
        
        
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
        return {
            'ard_weight_shape': list(self.ard_weights.shape),
            'is_force_cutoff': self.is_force_cutoff,
            'ratio_cutoff': self.ratio_cutoff,
            'heuristic_operation': self.heuristic_operation,
            'is_auto_adjust_gamma': self.is_auto_adjust_gamma,
            'is_dimension_median_heuristic': self.is_dimension_median_heuristic,
            'opt_bandwidth': self.opt_bandwidth
        }

    def _get_trainable_parameters(self) -> typing.List[torch.nn.Parameter]:
        return [self.ard_weights]

    def _get_median_single(self,
                           x: torch.Tensor,
                           y: torch.Tensor) -> torch.Tensor:
        """Get a median value for kernel functions.
        The approach is shown in 'Large sample analysis of the median heuristic'
        Args:
            x: (samples, features)
            y: (samples, features)
            minimum_sample: a minimum value for sampling.
            heuristic_operation: 'median' or 'mean'
        Returns:
            computed median
        """
        x_projected = torch.mul(self.ard_weights, x)  # elementwise product of ARD weight and x
        y_projected = torch.mul(self.ard_weights, y)  # elementwise product of ARD weight and y

        samp = torch.cat([x_projected, y_projected])
        np_reps = samp.detach().cpu().numpy()
        d2 = euclidean_distances(np_reps, squared=True)
        if self.heuristic_operation == 'median':
            med_sqdist = np.median(d2[np.triu_indices_from(d2, k=1)])
        elif self.heuristic_operation == 'mean':
            med_sqdist = np.mean(d2[np.triu_indices_from(d2, k=1)])
        else:
            raise Exception(f'No heuristic_operation == {self.heuristic_operation}.')
        # end if

        bandwidth = np.sqrt(med_sqdist / 2)

        if self.is_force_cutoff:
            bandwidth = self.adjust_auto_median_heuristic(x, y, bandwidth)
        # end if

        del samp, d2, med_sqdist
        # end if
        logger.debug("initial by median-heuristics {:.3g}".format(bandwidth))

        return torch.tensor([bandwidth])

    def _get_median_dim(self,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        is_completion_missing: bool = True,
                        is_safe_guard_same_xy: bool = True
                        ) -> typing.Optional[torch.Tensor]:
        """Get a median value for kernel functions.
        The approach is shown in 'Large sample analysis of the median heuristic'
        Args:
            x: (Samples, M)
            y: (Samples, M)
            ard_weight: ARD weight initialized.
        Returns:
            Median heuristic term with M where M is a dimension size.
        """
        with torch.no_grad():
            if self.distance_module.coordinate_size == 1:
                x_projected = torch.mul(self.ard_weights, x)  # elementwise product of ARD weight and x
                y_projected = torch.mul(self.ard_weights, y)  # elementwise product of ARD weight and y
            else:
                # comment: element-wise-product((N, |S|, C), (|S|))
                x_projected = torch.einsum('ijk,j->ijk', x, self.ard_weights)
                y_projected = torch.einsum('ijk,j->ijk', y, self.ard_weights)
                # double check multiplication.
                x_projected[0][0] = x[0][0] * self.ard_weights[0]
                y_projected[0][0] = y[0][0] * self.ard_weights[0]
                x_projected[-1][-1] = x[-1][-1] * self.ard_weights[-1]
                y_projected[-1][-1] = y[-1][-1] * self.ard_weights[-1]
            # end if
        # end with
                        
        # return None if all same.
        if is_safe_guard_same_xy:
            diff_xy = x_projected - y_projected
            # comment: all elements are zero. Hence returning None.
            if torch.count_nonzero(diff_xy) == 0:
                return None
            # end if
        # end if
        
        median_heuristic = torch.zeros((x.shape[1],))
        __shape = x_projected.shape
        n_dimension = x.shape[1]
        
        # ----------------------------------------------
        # arguments of task function
        __task_arguments = []
        for __m in range(0, n_dimension):
            if len(__shape) == 2:
                __x_projected_m = torch.reshape(x_projected[:, __m], shape=(len(x_projected), 1))
                __y_projected_m = torch.reshape(y_projected[:, __m], shape=(len(x_projected), 1))
            else:
                __x_projected_m = x_projected[:, __m]
                __y_projected_m = y_projected[:, __m]
            # end if
            __args_obj = DistributedFunctionArg(
                index_dimension=__m, 
                n_dimension=n_dimension, 
                x_projected_m=__x_projected_m,
                y_projected_m=__y_projected_m,
                heuristic_operation=self.heuristic_operation,
                distance_function=self.distance_module.compute_distance)
            __task_arguments.append(__args_obj)
        # end for
        # ----------------------------------------------
        # execution
        
        if self.dask_client is None:
            seq_length_scale = [compute_length_scale_dataset_dimension_d(__args) for __args in __task_arguments]
        else:
            dask_client = self.dask_client
            assert dask_client is not None, 'dask_client is None.' and isinstance(dask_client, Client)
            task_queue = dask_client.map(compute_length_scale_dataset_dimension_d, __task_arguments)
            seq_length_scale = dask_client.gather(task_queue)
        # end if
        # ----------------------------------------------
        for __tuple_length_scale in seq_length_scale:
            __index_dimension = __tuple_length_scale[0]
            __gamma_m = __tuple_length_scale[1]
            median_heuristic[__index_dimension] = __gamma_m
        # end for
        # ----------------------------------------------
        # post process
        res_value = torch.reshape(median_heuristic, self.ard_weights.shape)

        if self.distance_module.coordinate_size == 1:
            assert res_value.shape == x.shape[1:] == y.shape[1:]
        else:
            assert (res_value.shape[0], self.distance_module.coordinate_size) == x.shape[1:] == y.shape[1:]
        # end if

        if self.is_force_cutoff and self.ratio_cutoff == -1:
            ratio_cutoff = self.select_lower_bound_auto(x_projected, y_projected, res_value)
            self.ratio_cutoff = ratio_cutoff
        else:
            ratio_cutoff = self.ratio_cutoff
        # end if
        if self.is_force_cutoff:
            res_value = self.execute_force_cutoff(res_value, ratio_cutoff)
        # end if


        if is_completion_missing:
            # comment: when too few samples. Can be all 0.0. So, I skip the if block when too few samples.
            if len(x) < 50 and len(y) < 50:
                pass
            else:
                assert torch.count_nonzero(res_value) > 0, \
                    f'Kernel length scaling function encountered zero vales for all dimensions. Hint: changing kernel configuration to heuristic_operation="mean". Current heuristic_operation={self.heuristic_operation}'
                value_replace = torch.min(res_value[res_value != 0])
                res_value[res_value == 0] = value_replace
            # end if
        # endif
        return res_value.detach()

    def _compute_kernel_matrix_single(self,
                                      x: torch.Tensor,
                                      y: torch.Tensor,
                                      bandwidth: typing.Optional[torch.Tensor] = None
                                      ) -> KernelMatrixObject:
        # comment: I do not maintain this method anymore. Multi-dim length scale is the default.
        # Basically, I do not need this method anymore.
        x = torch.mul(self.ard_weights, x)
        y = torch.mul(self.ard_weights, y)

        if bandwidth is None:
            bandwidth = self.bandwidth
            assert bandwidth is not None
        # end if
        sigma = torch.exp(bandwidth)
        gamma = torch.div(1, (2 * torch.pow(sigma, 2)))

        # torch.t() is transpose function. torch.dot() is only for vectors. For 2nd tensors, "mm".
        # xx = torch.mm(x, torch.t(x))
        # xy = torch.mm(x, torch.t(y))
        # yy = torch.mm(y, torch.t(y))

        # x_sqnorms = torch.diagonal(xx, offset=0)
        # y_sqnorms = torch.diagonal(yy, offset=0)

        d_container = self.distance_module.compute_distance(x, y, False)

        k_xy = torch.exp(-1 * gamma * d_container.d_xy)
        k_xx = torch.exp(-1 * gamma * d_container.d_xx)
        k_yy = torch.exp(-1 * gamma * d_container.d_yy)

        k_container = QuadraticKernelMatrixContainer(k_xx, k_yy, k_xy)
        return KernelMatrixObject(kernel_computation_type=self.kernel_computation_type, x_size=len(x), y_size=len(y),
                                  kernel_matrix_container=k_container)

    def _compute_kernel_matrix_dim(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   bandwidth: typing.Optional[torch.Tensor] = None) -> KernelMatrixObject:
        if bandwidth is None:
            bandwidth = self.bandwidth
        # end if
        assert bandwidth is not None, 'bandwidth is None. Use set_length_scale(training_dataset) to set the bandwidth.'
        
        # # # Note: d_**_ard_gamma is L2-norm. Therefore, it should be square.
        # comment: previous implementation. Working correctly.       
        # x_ard = torch.mul(self.ard_weights, x)
        # y_ard = torch.mul(self.ard_weights, y)

        # x_ard_gamma = torch.div(x_ard, torch.sqrt(bandwidth))
        # y_ard_gamma = torch.div(y_ard, torch.sqrt(bandwidth))

        # __d_xx_ard_gamma = utils.distance_over_3rd_reshape_same_data(x_ard_gamma)
        # __d_yy_ard_gamma = utils.distance_over_3rd_reshape_same_data(y_ard_gamma)
        # __d_xy_ard_gamma = utils.distance_over_3rd_reshape_xy_data(x_ard_gamma, y_ard_gamma)

        # __d_xx_ard_gamma = __d_xx_ard_gamma.to(x.dtype)
        # __d_yy_ard_gamma = __d_yy_ard_gamma.to(x.dtype)

        # k_xx = torch.exp(-1 * torch.pow(__d_xx_ard_gamma, 2))
        # k_yy = torch.exp(-1 * torch.pow(__d_yy_ard_gamma, 2))
        # k_xy = torch.exp(-1 * torch.pow(__d_xy_ard_gamma, 2))
        # # end with

        if self.distance_module.coordinate_size == 1:
            x_projected = torch.mul(self.ard_weights, x)  # elementwise product of ARD weight and x
            y_projected = torch.mul(self.ard_weights, y)  # elementwise product of ARD weight and y            
            
            x_ard_gamma = torch.div(x_projected, torch.sqrt(bandwidth))
            y_ard_gamma = torch.div(y_projected, torch.sqrt(bandwidth))
        else:
            # comment: element-wise-product((N, |S|, C), (|S|))
            x_projected = torch.einsum('ijk,j->ijk', x, self.ard_weights)
            y_projected = torch.einsum('ijk,j->ijk', y, self.ard_weights)
            # double check multiplication.
            torch.testing.assert_close(x_projected[0][0].detach().to(torch.float32), expected=(x[0][0] * self.ard_weights[0]).detach().to(torch.float32))
            torch.testing.assert_close(y_projected[0][0].detach().to(torch.float32), expected=(y[0][0] * self.ard_weights[0]).detach().to(torch.float32))
            torch.testing.assert_close(x_projected[-1][-1].detach().to(torch.float32), expected=(x[-1][-1] * self.ard_weights[-1]).detach().to(torch.float32))
            torch.testing.assert_close(y_projected[-1][-1].detach().to(torch.float32), expected=(y[-1][-1] * self.ard_weights[-1]).detach().to(torch.float32))
            
            __gamma = torch.reciprocal(torch.sqrt(bandwidth))
            x_ard_gamma = torch.einsum('ijk,j->ijk', x_projected, __gamma)
            y_ard_gamma = torch.einsum('ijk,j->ijk', y_projected, __gamma)
        # end if
        
        d_container = self.distance_module.compute_distance(x_ard_gamma, y_ard_gamma, False)

        d_xx_ard_gamma = d_container.d_xx
        d_yy_ard_gamma = d_container.d_yy
        d_xy_ard_gamma = d_container.d_xy

        k_xx = torch.exp(-1 * d_xx_ard_gamma)
        k_yy = torch.exp(-1 * d_yy_ard_gamma)
        k_xy = torch.exp(-1 * d_xy_ard_gamma)
        
        k_container = QuadraticKernelMatrixContainer(k_xx, k_yy, k_xy)
        return KernelMatrixObject(kernel_computation_type=self.kernel_computation_type, x_size=len(x), y_size=len(y),
                                  kernel_matrix_container=k_container)


# TODO comment: this class is not used. 2023/11/15.
# LinearMMD does not show performance.
class LinearMMDGaussianKernel(QuadraticKernelGaussianKernel):
    """Implementation of Linear Gaussian described in EQ (4), Lemma 6 of "Kernel Two sample Test", Gretton, 2012.
    """
    def __init__(self,
                 distance_module: BaseDistanceModule = L2Distance(coordinate_size=1),
                 bandwidth: typing.Optional[torch.Tensor] = None,
                 ard_weights: typing.Optional[torch.Tensor] = None,
                 ard_weight_shape: typing.Optional[typing.Tuple[int, ...]] = None,
                 is_force_cutoff: bool = False,
                 ratio_cutoff: float = -1,
                 heuristic_operation: str = 'median',
                 is_auto_adjust_gamma: bool = False,
                 is_dimension_median_heuristic: bool = True,
                 opt_bandwidth: bool = False):
        super().__init__(
            distance_module=distance_module,
            bandwidth=bandwidth,
            ard_weights=ard_weights,
            ard_weight_shape=ard_weight_shape,
            is_force_cutoff=is_force_cutoff,
            ratio_cutoff=ratio_cutoff,
            heuristic_operation=heuristic_operation,
            is_auto_adjust_gamma=is_auto_adjust_gamma,
            is_dimension_median_heuristic=is_dimension_median_heuristic,
            opt_bandwidth=opt_bandwidth,
        )
        self.kernel_computation_type = 'linear'

    @classmethod
    def from_dataset(cls, 
                     dataset: BaseDataset,
                     distance_module: BaseDistanceModule = L2Distance(coordinate_size=1),
                     bandwidth: typing.Optional[torch.Tensor] = None,
                     ard_weights: typing.Optional[torch.Tensor] = None,
                     heuristic_operation: str = 'median',
                     is_dimension_median_heuristic: bool = True,
                     ) -> "LinearMMDGaussianKernel":
        """Public API. Create a kernel object from a dataset.
        """
        # do kernel length initialization.
        if ard_weights is None:
            __dim_x = dataset.get_dimension_data_space()
            ard_weights = torch.ones(__dim_x)
        # end if
        
        kernel_obj = cls(
            distance_module=distance_module,
            bandwidth=bandwidth,
            ard_weights=ard_weights,
            heuristic_operation=heuristic_operation,
            is_dimension_median_heuristic=is_dimension_median_heuristic)
        
        # do kernel length initialization.
        kernel_obj.compute_length_scale_dataset(dataset)
        
        return kernel_obj

    def _get_trainable_parameters(self) -> typing.List[torch.nn.Parameter]:
        return [self.ard_weights]

    def _get_median_single(self,
                           x: torch.Tensor,
                           y: torch.Tensor) -> torch.Tensor:
        """Get a median value for kernel functions.
        The approach is shown in 'Large sample analysis of the median heuristic'
        Args:
            x: (samples, features)
            y: (samples, features)
            minimum_sample: a minimum value for sampling.
            heuristic_operation: 'median' or 'mean'
        Returns:
            computed median
        """
        x_projected = torch.mul(self.ard_weights, x)  # elementwise product of ARD weight and x
        y_projected = torch.mul(self.ard_weights, y)  # elementwise product of ARD weight and y

        samp = torch.cat([x_projected, y_projected])
        diff = torch.pow(x_projected - y_projected, 2)
        if self.heuristic_operation == 'median':
            med_sqdist = torch.median(diff)
        elif self.heuristic_operation == 'mean':
            med_sqdist = torch.mean(diff)
        else:
            raise Exception(f'No heuristic_operation == {self.heuristic_operation}.')
        # end if

        bandwidth = torch.sqrt(med_sqdist / 2)

        if self.is_force_cutoff:
            bandwidth = self.adjust_auto_median_heuristic(x, y, bandwidth)
        # end if

        # end if
        logger.debug("initial by median-heuristics {:.3g}".format(bandwidth))
        assert len(bandwidth.shape) == 0, f"unexpected bandwidth shape. The bandwidth has {bandwidth.shape} size."

        return bandwidth

    def _get_median_dim(self,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        is_completion_missing: bool = True) -> typing.Optional[torch.Tensor]:
        """Get a median value for kernel functions.
        The approach is shown in 'Large sample analysis of the median heuristic'
        Args:
            x: (Samples, M)
            y: (Samples, M)
            ard_weight: ARD weight initialized.
        Returns:
            Median heuristic term with M where M is a dimension size.
        """
        x_projected = torch.mul(self.ard_weights, x)  # elementwise product of ARD weight and x
        y_projected = torch.mul(self.ard_weights, y)  # elementwise product of ARD weight and y
        # region computing median heuristic
        median_heuristic = torch.zeros((x.shape[1],))
        for d in range(0, x.shape[1]):
            x_dim = x_projected[:, d]  # (N, ) tensor
            y_dim = y_projected[:, d]  # (N, ) tensor

            m = len(x)

            d_xi_xj = utils.distance_over_3rd_reshape_xy_data(x_dim[:m:2], x_dim[1:m:2])
            d_yi_yj = utils.distance_over_3rd_reshape_xy_data(y_dim[:m:2], y_dim[1:m:2])
            d_xi_yj = utils.distance_over_3rd_reshape_xy_data(x_dim[:m:2], y_dim[1:m:2])
            d_xj_yi = utils.distance_over_3rd_reshape_xy_data(x_dim[1:m:2], y_dim[:m:2])

            if self.heuristic_operation == 'median':
                gamma_m: torch.Tensor = torch.median(torch.stack([d_xi_xj, d_yi_yj, d_xi_yj, d_xj_yi]))
            elif self.heuristic_operation == 'mean':
                gamma_m: torch.Tensor = torch.mean(torch.stack([d_xi_xj, d_yi_yj, d_xi_yj, d_xj_yi]))
            else:
                raise Exception(f'heuristic_operation == {self.heuristic_operation} does not exist.')
            # end if

            median_heuristic[d] = gamma_m * x.shape[1]
        # end for
        # endregion
        bandwidth = median_heuristic

        assert bandwidth.shape == x.shape[1:] == y.shape[1:]

        # if self.is_force_cutoff:
        #     # do auto-adjustment of bandwidth to realize acceptable K_xy values.
        #     bandwidth = self._replace_dim_wise_lower_bound(x_projected, y_projected, bandwidth)
        # # end if

        if is_completion_missing:
            value_replace = torch.min(bandwidth[bandwidth != 0])
            bandwidth[bandwidth == 0] = value_replace
        # endif
        return bandwidth.detach()

    def _compute_kernel_matrix_single(self,
                                      x: torch.Tensor,
                                      y: torch.Tensor,
                                      bandwidth: typing.Optional[torch.Tensor] = None) -> KernelMatrixObject:
        x = torch.mul(self.ard_weights, x)
        y = torch.mul(self.ard_weights, y)

        if bandwidth is None:
            bandwidth = self.bandwidth
            assert bandwidth is not None
        # end if

        sigma = torch.exp(bandwidth)
        gamma = torch.div(1, (2 * torch.pow(sigma, 2)))

        rbf = lambda A, B: torch.sum(torch.exp(-gamma * ((A - B) ** 2)), dim=1)

        n = (x.shape[0] // 2) * 2
        h_bits = (rbf(x[:n:2], x[1:n:2]) + rbf(y[:n:2], y[1:n:2])
                  - rbf(x[:n:2], y[1:n:2]) - rbf(x[1:n:2], y[:n:2]))

        k_container = LinearKernelMatrixContainer(h_bits)
        return KernelMatrixObject(kernel_computation_type=self.kernel_computation_type, x_size=len(x), y_size=len(y),
                                  kernel_matrix_container=k_container)

    def _compute_kernel_matrix_dim(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   bandwidth: typing.Optional[torch.Tensor] = None) -> KernelMatrixObject:
        """
        :param x:
        :param y:
        :param bandwidth:
        :return:
        """
        x = torch.mul(self.ard_weights, x)
        y = torch.mul(self.ard_weights, y)

        if bandwidth is None:
            bandwidth = self.bandwidth
        # end if

        assert bandwidth is not None, 'bandwidth is None. Use set_length_scale(training_dataset) to set the bandwidth.'
        x_ard_gamma = torch.div(x, torch.sqrt(bandwidth))
        y_ard_gamma = torch.div(y, torch.sqrt(bandwidth))

        m = len(x)

        d_xi_xj = utils.distance_over_3rd_reshape_xy_data(x_ard_gamma[:m:2], x_ard_gamma[1:m:2])
        d_yi_yj = utils.distance_over_3rd_reshape_xy_data(y_ard_gamma[:m:2], y_ard_gamma[1:m:2])
        d_xi_yj = utils.distance_over_3rd_reshape_xy_data(x_ard_gamma[:m:2], y_ard_gamma[1:m:2])
        d_xj_yi = utils.distance_over_3rd_reshape_xy_data(x_ard_gamma[1:m:2], x_ard_gamma[:m:2])

        k_xi_xj = torch.exp(-1 * torch.pow(d_xi_xj, 2))
        k_yi_yj = torch.exp(-1 * torch.pow(d_yi_yj, 2))
        k_xi_yi = torch.exp(-1 * torch.pow(d_xi_yj, 2))
        k_xj_yi = torch.exp(-1 * torch.pow(d_xj_yi, 2))

        h_bits = k_xi_xj + k_yi_yj - k_xi_yi - k_xj_yi
        k_container = LinearKernelMatrixContainer(h_bits)
        return KernelMatrixObject(kernel_computation_type=self.kernel_computation_type, x_size=len(x), y_size=len(y),
                                  kernel_matrix_container=k_container)
