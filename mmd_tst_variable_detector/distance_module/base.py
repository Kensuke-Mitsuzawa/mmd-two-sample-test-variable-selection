import abc
import typing

import torch
from dataclasses import dataclass


class DistanceContainer(typing.NamedTuple):
    """"""
    # Note: this class must be with NamedTuple.
    
    d_xx: torch.Tensor
    d_yy: torch.Tensor
    d_xy: torch.Tensor


class BaseDistanceModule(torch.nn.Module, metaclass=abc.ABCMeta):
    """Base class of distance module."""
    
    def __init__(self, data_point_shape: typing.Tuple[int, ...]) -> None:
        super(BaseDistanceModule, self).__init__()
        self.data_point_shape = data_point_shape
        # attribute
        self.coordinate_size: int

    @abc.abstractmethod
    def get_hyperparameters(self) -> typing.Dict[str, typing.Any]:
        pass

    @abc.abstractmethod
    def compute_distance(self, x: torch.Tensor, y: torch.Tensor, is_compute_length_scale: bool) -> DistanceContainer:
        """Compute distance between x and y."""
        raise NotImplementedError()
    
