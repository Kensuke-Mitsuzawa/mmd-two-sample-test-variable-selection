import typing as ty
import torch
from dataclasses import dataclass


# ------------------------------------------------------------
# inner states to used during inteference steps
# container to keep kernel matrix objects

@dataclass
class QuadraticKernelMatrixContainer:
    __slots__ = ('k_xx', 'k_yy', 'k_xy')
    
    k_xx: torch.Tensor
    k_yy: torch.Tensor
    k_xy: torch.Tensor

    def __post_init__(self):
        # k_xy is sometimes different data type.
        dtype_k_xx = self.k_xx.dtype
        self.k_xy = self.k_xy.to(dtype_k_xx)

@dataclass
class LinearKernelMatrixContainer:
    # See equation (4) of "A Kernel Two-Sample Test" Gretton, 2012
    __slots__ = ('k_h',)
    k_h: torch.Tensor


@dataclass
class KernelMatrixObject:
    kernel_computation_type: str  # 'quadratic', 'linear'
    x_size: int
    y_size: int
    kernel_matrix_container: ty.Union[LinearKernelMatrixContainer, QuadraticKernelMatrixContainer]

