from pathlib import Path

import torch
import torch.utils.data

from mmd_tst_variable_detector.datasets.ram_backend_static_dataset import SimpleDataset
from mmd_tst_variable_detector.kernels.kronecker_delta_kernel import QuadraticKronekerDeltaKernel

from .. import data_generator


def test_QuadraticKernelGaussianKernel(resource_path_root: Path):
    t_xy, __ = data_generator.test_data_discrete_category()
    my_dataset = SimpleDataset(t_xy[0], t_xy[1])

    initial_ard = torch.ones(my_dataset.get_dimension_flattened())

    kernel = QuadraticKronekerDeltaKernel(ard_weights=initial_ard)
    loader = torch.utils.data.DataLoader(my_dataset, batch_size=100)
    for x, y in loader:
        kernel.compute_kernel_matrix(x, y)
