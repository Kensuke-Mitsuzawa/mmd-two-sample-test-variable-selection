import torch
import typing as ty
import typing

from .base import BaseKernel
from .commons import (KernelMatrixObject, QuadraticKernelMatrixContainer)

from ..datasets import BaseDataset


class MaternKernel(BaseKernel):
    """Matern kernel without ARD weights, forms of 0.5, 1.5, 2.5. 
    """
    def __init__(self,
                 length_scale: float =1.0, 
                 nu: float = 1.5
                 #  ard_weight_shape: typing.Optional[typing.Tuple[int, ...]],):
                 ):
        # note: `ard_weight_shape` is a dummy variable. It's not used.
        super().__init__()
        assert nu in (0.5, 1.5, 2.5), f'nu parameter must be either of {(0.5, 1.5, 2.5)}'
        self.length_scale = length_scale
        self.nu = nu

    def from_dataset(cls, dataset: BaseDataset) -> "MaternKernel":
        """Public API method to create a kernel object from a dataset.
        
        Must be implemented in a subclass.
        """
        dim_shape = dataset.get_dimension_flattened()
        return MaternKernel()

    # --------------------------------------------------------------------------------------------------
    # methods to be implemented.
    # private methods

    def _get_median_single(self,
                           x: torch.Tensor,
                           y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _get_median_dim(self,
                        x: torch.Tensor,
                        y: torch.Tensor) -> typing.Optional[torch.Tensor]:
        raise NotImplementedError()

    def _compute_kernel_matrix_single(self,
                                      x: torch.Tensor,
                                      y: torch.Tensor,
                                      bandwidth: typing.Optional[torch.Tensor]) -> KernelMatrixObject:
        raise NotImplementedError()

    def _compute_kernel_matrix_dim(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   bandwidth: typing.Optional[torch.Tensor]) -> KernelMatrixObject:
        raise NotImplementedError()

    def _get_trainable_parameters(self) -> typing.List[torch.nn.Parameter]:
        """An abstract method that returns a list of trainable parameters.

        :return:
        """
        raise NotImplementedError()
    
    # -----------------------------------------------------------------------------    
    # methods to be implemented.
    # public API methods

    def get_hyperparameters(self) -> typing.Dict[str, typing.Any]:
        """A method to return a dictionary of hyperparameters.

        :return:
        """
        raise NotImplementedError()

    def matern_kernel(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Matern Kernel.
        
        Args:
            x1 (torch.Tensor): A tensor of shape (N, D).
            x2 (torch.Tensor): A tensor of shape (M, D).
            length_scale (float): The length scale hyperparameter.
            nu (float): The smoothness parameter (e.g., 0.5, 1.5, 2.5).
            
        Returns:
            torch.Tensor: The kernel matrix of shape (N, M).
        """
        # Euclidean distance
        dist = torch.cdist(x1, x2, p=2)
        dist_scaled = dist / self.length_scale
        
        if self.nu == 0.5:
            k = torch.exp(-dist_scaled)
        elif self.nu == 1.5:
            sqrt3_dist = torch.sqrt(torch.tensor(3.0)) * dist_scaled
            k = (1.0 + sqrt3_dist) * torch.exp(-sqrt3_dist)
        elif self.nu == 2.5:
            sqrt5_dist = torch.sqrt(torch.tensor(5.0)) * dist_scaled
            k = (1.0 + sqrt5_dist + (5.0/3.0) * dist_scaled**2) * torch.exp(-sqrt5_dist)
        else:
            raise ValueError("Matern kernel is only implemented for nu=0.5, 1.5, or 2.5.")
            
        return k

    def compute_kernel_matrix(self,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              bandwidth: ty.Optional[torch.Tensor] = None) -> KernelMatrixObject:
        k_xx = self.matern_kernel(x, x)
        k_yy = self.matern_kernel(y, y)
        k_xy = self.matern_kernel(x, y)

        k_obj = KernelMatrixObject(
            kernel_computation_type='quadratic',
            x_size=x.shape[0],
            y_size=y.shape[0],
            kernel_matrix_container=QuadraticKernelMatrixContainer(
                k_xx=k_xx,
                k_yy=k_yy,
                k_xy=k_xy
            )
        )
        return k_obj