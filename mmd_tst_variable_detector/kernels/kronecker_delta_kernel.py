import typing as ty

import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from ..datasets.base import BaseDataset
from .base import BaseKernel, KernelMatrixObject
from .commons import QuadraticKernelMatrixContainer
from .utils import distance_over_3rd_reshape_same_data, distance_over_3rd_reshape_xy_data
from ..distance_module import L2Distance

class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)


class QuadraticKronekerDeltaKernel(BaseKernel):
    def __init__(self, ard_weights: torch.Tensor):
        super().__init__(
            distance_module=L2Distance(coordinate_size=1),
            possible_shapes=(1, 2),
            ard_weights=ard_weights,
            kernel_computation_type='quadratic',
            is_dimension_median_heuristic=True
        )
        
    @classmethod
    def from_dataset(cls, dataset: BaseDataset) -> "QuadraticKronekerDeltaKernel":
        __dim_x, __dim_y = dataset.get_dimension_flattened()
        return cls(ard_weights=torch.ones(__dim_x))
        
    def get_hyperparameters(self) -> ty.Dict[str, ty.Any]:
        return {
            'kernel_computation_type': self.kernel_computation_type,
            'is_dimension_median_heuristic': self.is_dimension_median_heuristic
        }        

    def _get_trainable_parameters(self) -> ty.List[torch.nn.Parameter]:
        return [self.ard_weights]

    def _get_median_single(self,
                           x: torch.Tensor,
                           y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This kernel does not have the length scale.")

    def _get_median_dim(self,
                        x: torch.Tensor,
                        y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This kernel does not have the length scale.")

    def _compute_kernel_matrix_single(self,
                                      x: torch.Tensor,
                                      y: torch.Tensor,
                                      bandwidth: ty.Optional[torch.Tensor] = None) -> KernelMatrixObject:
        return self._compute_kernel_matrix_dim(x, y, bandwidth)

    def _compute_kernel_matrix_dim(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   bandwidth: ty.Optional[torch.Tensor] = None
                                   ) -> KernelMatrixObject:
        # discrete kernel
        x_part_discrete = x
        y_part_discrete = y

        x_part_discrete_ard = torch.mul(self.ard_weights, x_part_discrete)
        y_part_discrete_ard = torch.mul(self.ard_weights, y_part_discrete)

        if self.kernel_computation_type == 'linear':
            raise NotImplementedError('Not implemented yet.')
        elif self.kernel_computation_type == 'quadratic':
            dist_xx = 1.0 - distance_over_3rd_reshape_same_data(x_part_discrete_ard)
            dist_yy = 1.0 - distance_over_3rd_reshape_same_data(y_part_discrete_ard)
            dist_xy = 1.0 - distance_over_3rd_reshape_xy_data(x_part_discrete_ard, y_part_discrete_ard)
            kronecker_xx = dclamp(dist_xx, 1e-5, 1.0)
            kronecker_yy = dclamp(dist_yy, 1e-5, 1.0)
            kronecker_xy = dclamp(dist_xy, 1e-5, 1.0)

            return KernelMatrixObject(
                kernel_computation_type=self.kernel_computation_type,
                x_size=len(x),
                y_size=len(y),
                kernel_matrix_container=QuadraticKernelMatrixContainer(
                    k_xx=kronecker_xx,
                    k_yy=kronecker_yy,
                    k_xy=kronecker_xy)
            )
        else:
            raise NotImplementedError()
