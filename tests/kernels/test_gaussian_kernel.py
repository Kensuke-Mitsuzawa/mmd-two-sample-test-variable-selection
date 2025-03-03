from pathlib import Path

import torch
import torch.utils.data

from distributed import Client, LocalCluster

from mmd_tst_variable_detector.datasets.ram_backend_static_dataset import SimpleDataset
from mmd_tst_variable_detector.kernels.gaussian_kernel import (
    LinearMMDGaussianKernel,
    QuadraticKernelGaussianKernel)
from mmd_tst_variable_detector.distance_module.l2_distance import L2Distance

from .. import data_generator


def test_QuadraticKernelGaussianKernel_length_scale_single_thread(resource_path_root: Path):
    """Test the length scale computation of the QuadraticKernelGaussianKernel."""
    t_xy, __ = data_generator.test_data_xy_linear()
    my_dataset = SimpleDataset(t_xy[0], t_xy[1])

    initial_ard = torch.ones(my_dataset.get_dimension_flattened())

    distance_module = L2Distance(coordinate_size=1)

    kernel = QuadraticKernelGaussianKernel(distance_module=distance_module, ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(my_dataset)
    kernel.set_length_scale()
    loader = torch.utils.data.DataLoader(my_dataset, batch_size=100)
    for x, y in loader:
        kernel.compute_kernel_matrix(x, y)


def test_QuadraticKernelGaussianKernel_length_dask(resource_path_root: Path):
    """Test the length scale computation of the QuadraticKernelGaussianKernel.
    I do comparison of bandwidth computing between single loop and dask."""
    t_xy, __ = data_generator.test_data_xy_linear()
    my_dataset = SimpleDataset(t_xy[0], t_xy[1])

    initial_ard = torch.ones(my_dataset.get_dimension_flattened())

    dask_cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    dask_client = Client(dask_cluster)

    distance_module = L2Distance(coordinate_size=1)

    kernel = QuadraticKernelGaussianKernel(distance_module=distance_module, 
                                           ard_weights=initial_ard, 
                                           dask_client=dask_client)
    kernel.compute_length_scale_dataset(my_dataset)
    kernel.set_length_scale()
    loader = torch.utils.data.DataLoader(my_dataset, batch_size=100)
    for x, y in loader:
        kernel.compute_kernel_matrix(x, y)
    # end for
    
    dask_client.close()
    dask_cluster.close()
    del dask_client
    del dask_cluster
    
    # comparison with the single mode
    kernel_single = QuadraticKernelGaussianKernel.from_dataset(my_dataset)
    
    # check length-scales
    assert torch.eq(kernel.bandwidth, kernel_single.bandwidth).all()
    

def test_LinearGaussianKernel(resource_path_root: Path):
    t_xy, __ = data_generator.test_data_xy_linear()
    my_dataset = SimpleDataset(t_xy[0], t_xy[1])

    initial_ard = torch.ones(my_dataset.get_dimension_flattened())

    distance_module = L2Distance(coordinate_size=1)

    kernel = LinearMMDGaussianKernel(distance_module, ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(my_dataset)
    kernel.set_length_scale()
    loader = torch.utils.data.DataLoader(my_dataset, batch_size=100)
    for x, y in loader:
        kernel.compute_kernel_matrix(x, y)
