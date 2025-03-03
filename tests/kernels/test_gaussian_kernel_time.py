from pathlib import Path

import torch
import torch.utils.data

from distributed import Client, LocalCluster

from mmd_tst_variable_detector.datasets.ram_backend_static_dataset import SimpleDataset
from mmd_tst_variable_detector.kernels.gaussian_kernel_time import TimeAwareQuadraticKernelGaussianKernel
from mmd_tst_variable_detector.distance_module.time_aware_distance import TimeAwareDistance

from .. import data_generator


def __generate_time_aware_distance(n_sample: int = 5,
                                   n_timstamp: int = 5,
                                   n_sensor: int = 10):
    """Generate a TimeAwareDistance object. Pseudo data is generated."""
    set_x = torch.zeros((n_sample, n_sensor, n_timstamp))
    set_y = torch.zeros((n_sample, n_sensor, n_timstamp))
    
    for __i_sample in range(n_sample):
        x = torch.randn(n_sensor, n_timstamp)
        y = torch.randn(n_sensor, n_timstamp)
        
        set_x[__i_sample] = x
        set_y[__i_sample] = y
    # end for
    return set_x, set_y
    


def test_QuadraticKernelGaussianKernel_single_loop(resource_path_root: Path):
    """Test the length scale computation of the QuadraticKernelGaussianKernel.
    I do comparison of bandwidth computing between single loop and dask."""
    set_x, set_y = __generate_time_aware_distance()
    my_dataset = SimpleDataset(set_x, set_y)

    initial_ard = torch.ones(my_dataset.get_dimension_flattened())

    dask_client = None

    distance_module = TimeAwareDistance(coordinate_size=1)

    n_sensors = set_x.shape[1]
    n_timestamp = set_x.shape[2]

    kernel = TimeAwareQuadraticKernelGaussianKernel(distance_module=distance_module, 
                                                    n_sensors=n_sensors,
                                                    n_timestamp=n_timestamp)
    kernel.compute_length_scale_dataset(my_dataset)
    kernel.set_length_scale()
    loader = torch.utils.data.DataLoader(my_dataset, batch_size=100)
    for x, y in loader:
        kernel.compute_kernel_matrix(x, y)
    # end for
    