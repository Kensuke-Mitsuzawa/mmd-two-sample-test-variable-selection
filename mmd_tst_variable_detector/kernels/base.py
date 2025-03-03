import abc
import types
import typing
from dataclasses import dataclass
import logging

import torch
import torch.utils.data
import numpy as np
from torch.nn import Parameter

from ..distance_module.base import BaseDistanceModule, DistanceContainer
from ..datasets.base import BaseDataset
from .commons import KernelMatrixObject

from ..exceptions import SameDataException

from ..logger_unit import handler
kernel_module_logger = logging.getLogger(f'{__package__}.kernels')
kernel_module_logger.addHandler(handler)


DEFAULT_RATIO_CUTOFF_CANDIDATES = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.7, 0.9) + tuple(
    list(np.arange(1, 10, step=0.5)))


# ------------------------------------------------------------------------------
# Base kernel object


class BaseKernel(torch.nn.Module, metaclass=abc.ABCMeta):
    """Base class of Kernel function class."""

    def __init__(self,
                 distance_module: BaseDistanceModule,
                 possible_shapes: typing.Tuple[int, ...],
                 bandwidth: typing.Optional[torch.Tensor] = None,
                 ard_weights: typing.Optional[torch.Tensor] = None,
                 ard_weight_shape: typing.Optional[typing.Tuple[int, ...]] = None,
                 is_force_cutoff: bool = False,
                 ratio_cutoff: float = -1,
                 heuristic_operation: str = 'median',
                 is_auto_adjust_gamma: bool = False,
                 is_dimension_median_heuristic: bool = True,
                 opt_bandwidth: bool = False,
                 kernel_computation_type: str = "quadratic"):
        super(BaseKernel, self).__init__()
        
        self.distance_module = distance_module
        
        self.possible_shapes = possible_shapes
        self.bandwidth = bandwidth

        self.is_dimension_median_heuristic = is_dimension_median_heuristic
        self.is_force_cutoff = is_force_cutoff
        self.ratio_cutoff = ratio_cutoff
        self.heuristic_operation = heuristic_operation
        self.is_auto_adjust_gamma = is_auto_adjust_gamma
        self.opt_bandwidth = opt_bandwidth
        self.kernel_computation_type = kernel_computation_type

        if ratio_cutoff != -1:
            assert ratio_cutoff > 0.0, 'ratio_cutoff should be > 0.0.'
        # end if

        if ard_weights is not None:
            assert isinstance(ard_weights, torch.Tensor)
            self.ard_weights = Parameter(ard_weights)
        else:
            assert ard_weight_shape is not None, 'ard_weight_shape must be given.'
            self.ard_weights = Parameter(torch.ones(ard_weight_shape))

        if bandwidth is not None:
            self.bandwidth = Parameter(bandwidth, requires_grad=False)
        else:
            self.bandwidth = Parameter(torch.ones(self.ard_weights.shape), requires_grad=False)
        # end if

        self.__stack_bandwidth = []

    @classmethod
    @abc.abstractmethod
    def from_dataset(cls, dataset: BaseDataset) -> "BaseKernel":
        """Public API method to create a kernel object from a dataset.
        
        Must be implemented in a subclass.
        """
        raise NotImplementedError()
    

    def reset_variables(self):
        self.__stack_bandwidth = []

    def check_data_shape(self, data: torch.Tensor):
        if len(data.shape) not in self.possible_shapes:
            raise Exception(f'Input data has {len(data.shape)} tensor. '
                            f'But the kernel class expects {self.possible_shapes} tensor.')

    def select_lower_bound_auto(self,
                                x_projected: torch.Tensor,
                                y_projected: torch.Tensor,
                                gamma_matrix: torch.Tensor,
                                avg_k_xy_threshold: float = 0.4,
                                default_ratio_cutoff_candidates: typing.Sequence[
                                    float] = DEFAULT_RATIO_CUTOFF_CANDIDATES
                                ) -> float:
        """Automatically choosing the appropriate bandwidth values for the dimension wise **-heuristic.
        Args:
            x_projected: X samples projected by ARD weights.
            y_projected: Y samples projected by ARD weights.
            gamma_matrix: Bandwidth. A vector form or a matrix form.
            avg_k_xy_threshold: A threshold value to pass the condition. A hyperparameter.
            default_ratio_cutoff_candidates:
        Returns:
        """
        stack = []
        for __ratio_cutoff in default_ratio_cutoff_candidates:
            # force cut off
            cutoff_threshold = gamma_matrix.max() * __ratio_cutoff
            interpolate_values = torch.mean(torch.tensor([cutoff_threshold, gamma_matrix.max()]))
            gamma_matrix[gamma_matrix < cutoff_threshold] = interpolate_values
            kernel_module_logger.debug(f"Applied is_force_cutoff option. "
                         f"In gamma tensor, values less than {cutoff_threshold} are replaced with {interpolate_values}.")
            __bandwidth = gamma_matrix.clone().detach()
            __k_obj = self.compute_kernel_matrix(x=x_projected, y=y_projected, bandwidth=__bandwidth)
            if __k_obj.kernel_computation_type == 'quadratic':
                avg_k_value = torch.mean(__k_obj.kernel_matrix_container.k_xy)
            elif __k_obj.kernel_computation_type == 'linear':
                avg_k_value = torch.mean(__k_obj.kernel_matrix_container.k_h)
            else:
                raise NotImplementedError()
            # end if

            stack.append([__ratio_cutoff, avg_k_value])

            if avg_k_value > avg_k_xy_threshold:
                return __ratio_cutoff
        # end for
        ratio_cutoff_biggest = list(sorted(stack, key=lambda t: t[1], reverse=True))
        kernel_module_logger.warn(
            f'select_lower_bound_auto() failed to set an appropriate bandwidth. The mean(K_xy) is finally {ratio_cutoff_biggest[0][1]}. '
            'When optimizations does not advance at all, the computed bandwidth is probably the factor. '
            'In that case, try to set a large value to the argument `default_ratio_cutoff_candidates`.')
        return ratio_cutoff_biggest[0][0]

    @staticmethod
    def execute_force_cutoff(gamma: torch.Tensor, ratio_cutoff: float) -> torch.Tensor:
        """Replace small values in the gamma matrix with a lower threshold.
        A lower threshold is chosen by `ratio_cutoff`.
        Args:
            gamma: `torch.Tensor`. A gamma matrix in math:`D \times T`.
            ratio_cutoff: `float`
        Returns: `torch.Tensor`. A gamma matrix in math:`D \times T`.
        """
        cutoff_threshold = gamma.max() * ratio_cutoff
        interpolate_values = torch.mean(torch.tensor([cutoff_threshold, gamma.max()]))
        gamma[gamma < cutoff_threshold] = interpolate_values
        kernel_module_logger.debug(f"Applied is_force_cutoff option. "
                                   f"In gamma tensor, values less than {cutoff_threshold} are replaced with {interpolate_values}.")
        return gamma

    def adjust_auto_median_heuristic(self,
                                     x: torch.Tensor,
                                     y: torch.Tensor,
                                     bandwidth: torch.Tensor,
                                     avg_k_xy_lower: float = 0.1,
                                     avg_k_xy_upper: float = 0.8) -> torch.Tensor:
        """Selecting a median heuristic value that Kernel Matrix K_XY can have a reasonable value.
        Args:
            x: a product of ARD-weights and sample-x.
            y: a product of ARD-weights and sample-y.
            bandwidth: an initial value of bandwidth.
            avg_k_xy_upper: a threshold value of the lower bound
            avg_k_xy_lower: a threshold value of the upper bound
        Returns: A selected bandwidth
        """
        def __kernel_value_criteria(k_object: KernelMatrixObject) -> bool:
            if self.kernel_computation_type == 'quadratic':
                k_matrix = k_object.kernel_matrix_container.k_xy
            elif self.kernel_computation_type == 'linear':
                k_matrix = k_object.kernel_matrix_container.k_h
            else:
                raise NotImplementedError()
            # end if

            if self.kernel_computation_type == 'quadratic':
                if avg_k_xy_lower < k_matrix.mean() < avg_k_xy_upper:
                    return True
            elif self.kernel_computation_type == 'linear':
                mmd2 = k_object.kernel_matrix_container.k_h.mean()
                if mmd2.item() > 0.0:
                    return True
            else:
                raise Exception()
            
            return False
        # end if

        if self.kernel_computation_type == 'quadratic':
            adjustment_candidate = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001] + list(np.arange(1, 100, 10))
        elif self.kernel_computation_type == 'linear':
            adjustment_candidate = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001] + list(np.arange(1, 100, 1))
        else:
            raise Exception()
        # end if

        x_projected = torch.mul(self.ard_weights, x)
        y_projected = torch.mul(self.ard_weights, y)

        k_object: KernelMatrixObject = self.compute_kernel_matrix(x=x_projected,
                                                                  y=y_projected,
                                                                  bandwidth=bandwidth.clone().detach())

        is_pass = __kernel_value_criteria(k_object)
        if is_pass:
            return bandwidth
        # end if

        for adj_value in adjustment_candidate:
            bandwidth_candidate = bandwidth * adj_value
            k_object = self.compute_kernel_matrix(x=x_projected, y=y_projected, bandwidth=torch.tensor(bandwidth_candidate))  # noqa
            is_pass = __kernel_value_criteria(k_object)
            if is_pass:
                return bandwidth_candidate
            # end if
        # end for
        kernel_module_logger.warning("Failed to adjust the bandwidth value. Probably, a calculated MMD values is not appropriate.")
        return bandwidth

    def set_length_scale(self, agg_func: str = 'mean') -> None:
        """A method to set the bandwidth parameter. self.__stack_bandwidth should be computed beforehand.
        :param agg_func:
        :return:
        """
        if len(self.__stack_bandwidth) == 0:
            raise SameDataException(
                'A stack of the batched bandwidth is null.' 
                'Either your data pairs XY are totally same or you forgot calling `iter_call_length_scale()` first.')
        # end if

        # aggregation of the computed bandwidth
        if self.is_dimension_median_heuristic:
            __dim_parameter = 0
        else:
            __dim_parameter = None
        # end if
        if agg_func == 'mean':
            bandwidth = torch.mean(torch.stack(self.__stack_bandwidth), dim=__dim_parameter)
        else:
            raise NotImplementedError('Not implemented yet.')
        # end if

        __bandwidth = bandwidth.clone().detach()
        if self.opt_bandwidth is True:
            self.bandwidth = Parameter(__bandwidth.requires_grad_(True))
        else:
            self.bandwidth = Parameter(__bandwidth, requires_grad=False)

    def iter_call_length_scale(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """A method to be called during bandwidth computing. This method adds a batched bandwidth value to a stack (list).
        :param x:
        :param y:
        :return: None.
        """
        assert torch.isnan(x).sum() == 0, 'input data tensor has nan values.'
        assert torch.isnan(y).sum() == 0, 'input data tensor has nan values.'

        if self.is_dimension_median_heuristic:
            __b = self._get_median_dim(x, y)
        else:
            __b = self._get_median_single(x, y)
        # end if
        
        if __b is None:
            kernel_module_logger.warning('X and Y are all same. Skip this iteration.')
            return None

        if self.is_auto_adjust_gamma:
            self.adjust_auto_median_heuristic(x, y, bandwidth=__b)
        
        self.__stack_bandwidth.append(__b.detach())

    def compute_length_scale_dataset(self, dataset: BaseDataset, batch_size: int = 100, agg_func: str = 'mean'):
        if dataset.is_dataset_on_ram():
            dataset = dataset.generate_dataset_on_ram()
        # end if
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        for x, y in loader:
            if torch.count_nonzero(x - y) == 0:
                kernel_module_logger.warning('X and Y are all same. Skip this iteration.')
                continue
            # end if
            __x = x.detach()
            __y = y.detach()
            self.iter_call_length_scale(__x, __y)
        # end for
        self.set_length_scale(agg_func=agg_func)

    def compute_kernel_matrix(self,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              bandwidth: typing.Optional[torch.Tensor] = None) -> KernelMatrixObject:
        if self.is_dimension_median_heuristic:
            return self._compute_kernel_matrix_dim(x, y, bandwidth)
        else:
            return self._compute_kernel_matrix_single(x, y, bandwidth)

    # --------------------------------------------------------------------------------------------------
    # methods to be implemented.
    # private methods

    @abc.abstractmethod
    def _get_median_single(self,
                           x: torch.Tensor,
                           y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_median_dim(self,
                        x: torch.Tensor,
                        y: torch.Tensor) -> typing.Optional[torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _compute_kernel_matrix_single(self,
                                      x: torch.Tensor,
                                      y: torch.Tensor,
                                      bandwidth: typing.Optional[torch.Tensor]) -> KernelMatrixObject:
        raise NotImplementedError()

    @abc.abstractmethod
    def _compute_kernel_matrix_dim(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   bandwidth: typing.Optional[torch.Tensor]) -> KernelMatrixObject:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_trainable_parameters(self) -> typing.List[torch.nn.Parameter]:
        """An abstract method that returns a list of trainable parameters.

        :return:
        """
        raise NotImplementedError()
    
    # -----------------------------------------------------------------------------    
    # methods to be implemented.
    # public API methods

    @abc.abstractmethod
    def get_hyperparameters(self) -> typing.Dict[str, typing.Any]:
        """A method to return a dictionary of hyperparameters.

        :return:
        """
        raise NotImplementedError()

# -----------------------------------------------------------------------------
