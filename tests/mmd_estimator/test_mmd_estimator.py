from pathlib import Path
import json
from dataclasses import asdict
import torch
import torch.utils.data

from mmd_tst_variable_detector.datasets.ram_backend_static_dataset import SimpleDataset
from mmd_tst_variable_detector.kernels.gaussian_kernel import (
    LinearMMDGaussianKernel,
    QuadraticKernelGaussianKernel)
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator, LinearMmdEstimator, MmdValues

from tests import data_generator


def test_QuadraticMmdEstimator(resource_path_root: Path):
    t_xy, __ = data_generator.test_data_xy_linear()
    my_dataset = SimpleDataset(t_xy[0], t_xy[1])

    initial_ard = torch.ones(my_dataset.get_dimension_flattened())

    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(my_dataset)
    kernel.set_length_scale()

    mmd_estimator = QuadraticMmdEstimator(kernel)
    
    # -----------------------------------------------------
    # testing `from_dataset` method
    mmd_estimator = QuadraticMmdEstimator.from_dataset(my_dataset, kernel_class=QuadraticKernelGaussianKernel)
    
    loader = torch.utils.data.DataLoader(my_dataset, batch_size=100)
    for x, y in loader:
        mmd_obj = mmd_estimator.forward(x, y)
        isinstance(mmd_obj, MmdValues)
    # end for
    
    # saving parameters and reloading estimator
    arg_parameters = mmd_estimator.get_hyperparameters()
    json.dumps(asdict(arg_parameters))
    parameter_dicts = mmd_estimator.state_dict()
    
    __kernel = QuadraticKernelGaussianKernel(**arg_parameters.kernel_object_arguments)
    mmd_estimator = QuadraticMmdEstimator(kernel_obj=__kernel, **arg_parameters.mmd_object_arguments)
    mmd_estimator.load_state_dict(parameter_dicts)


def test_LinearHsicMmdEstimator(resource_path_root: Path):
    t_xy, __ = data_generator.test_data_xy_linear()
    my_dataset = SimpleDataset(t_xy[0], t_xy[1])

    initial_ard = torch.ones(my_dataset.get_dimension_flattened())

    kernel = LinearMMDGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(my_dataset)
    kernel.set_length_scale()

    mmd_estimator = LinearMmdEstimator(kernel)
    # -----------------------------------------------------
    # testing `from_dataset` method
    
    mmd_estimator = LinearMmdEstimator.from_dataset(my_dataset, kernel_class=LinearMMDGaussianKernel)

    loader = torch.utils.data.DataLoader(my_dataset, batch_size=100)
    for x, y in loader:
        mmd_obj = mmd_estimator.forward(x, y)
        isinstance(mmd_obj, MmdValues)
    # end for
    
    # saving parameters and reloading estimator
    arg_parameters = mmd_estimator.get_hyperparameters()
    json.dumps(asdict(arg_parameters))
    parameter_dicts = mmd_estimator.state_dict()
    
    __kernel = LinearMMDGaussianKernel(**arg_parameters.kernel_object_arguments)
    mmd_estimator = LinearMmdEstimator(kernel_obj=__kernel, **arg_parameters.mmd_object_arguments)
    mmd_estimator.load_state_dict(parameter_dicts)
