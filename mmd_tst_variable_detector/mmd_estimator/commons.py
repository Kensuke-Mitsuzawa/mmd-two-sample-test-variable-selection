import typing as ty
import torch
from ..kernels.commons import KernelMatrixObject
from dataclasses import dataclass


@dataclass
class MmdValues:
    # Comment: this class must be with NamedTuple, for performance and maintainability.
    
    mmd: torch.Tensor
    variance: torch.Tensor
    ratio: ty.Optional[torch.Tensor]
    kernel_matrix_obj: ty.Optional[KernelMatrixObject] = None


@dataclass
class ArgumentParameters:
    __slots__ = ('distance_class_name', 'kernel_class_name', 'mmd_estimator_class_name',
                 'distance_object_arguments', 'kernel_object_arguments', 'mmd_object_arguments',
                 'package_version')
    
    distance_class_name: str
    kernel_class_name: str
    mmd_estimator_class_name: str
    distance_object_arguments: ty.Dict[str, ty.Any]
    kernel_object_arguments: ty.Dict[str, ty.Any]
    mmd_object_arguments: ty.Dict[str, ty.Any]
    package_version: str
