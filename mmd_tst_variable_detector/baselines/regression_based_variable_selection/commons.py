import torch
import geomloss
import functools
from ot import sliced_wasserstein_distance
from ...utils import PermutationTest


geomloss_obj = geomloss.SamplesLoss(loss='sinkhorn')


def _base_func_distance_sinkhorn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return geomloss_obj(x, y)


def _base_func_distance_sliced_wasserstein(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.Tensor(sliced_wasserstein_distance(x.numpy(), y.numpy()))
