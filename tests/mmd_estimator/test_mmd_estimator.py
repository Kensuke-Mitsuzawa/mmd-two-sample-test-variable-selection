from pathlib import Path
import json
from dataclasses import asdict
import torch
import torch.utils.data

from mmd_tst_variable_detector.datasets.ram_backend_static_dataset import (RamBackendStaticDataset, SimpleDataset)
from mmd_tst_variable_detector.kernels.gaussian_kernel import (
    LinearMMDGaussianKernel,
    QuadraticKernelGaussianKernel)
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator, LinearMmdEstimator, MmdValues

from tests import data_generator

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_QuadraticMmdEstimator_case_one_sample(resource_path_root: Path):
    """Computing the MMD value when the sample size is 1 (setting actually 2)"""
    t_xy_train, __ = data_generator.test_data_xy_linear(random_seed=1234)
    
    t_xy_calibration, __ = data_generator.test_data_xy_linear(random_seed=42)
    dataset_calibration = RamBackendStaticDataset(t_xy_calibration[0], t_xy_calibration[1])

    initial_ard = torch.ones(dataset_calibration.get_dimension_flattened())

    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(dataset_calibration)
    kernel.set_length_scale()

    mmd_estimator = QuadraticMmdEstimator(kernel,
                                          variance_term="sutherland_2017",
                                          biased=False,
                                          unit_diagonal=False)
    
    # I copy the same sample
    sample_x_small = torch.tensor([t_xy_train[0][0].tolist()] * 2, dtype=torch.float64)
    sample_x_big = torch.tensor([t_xy_train[0][0].tolist()] * 1000, dtype=torch.float64)
    # -----------------------------------------------------
    # X: 3 samples, Y: 1000 samples
    
    result_small_sample = mmd_estimator.forward(x=sample_x_small, y=t_xy_train[1], is_add_kernel_matrix_object=True)
    # -----------------------------------------------------
    # X: 1000 samples, Y: 1000 samples
    result_big_sample = mmd_estimator.forward(x=sample_x_big, y=t_xy_train[1], is_add_kernel_matrix_object=True)

    mmd_small_sample = result_small_sample.mmd.item()
    mmd_big_sample = result_big_sample.mmd.item()

    logger.debug(f"mmd_result_small_sample: {mmd_small_sample}, mmd_result_full_sample: {mmd_big_sample}")
    assert abs(mmd_small_sample - mmd_big_sample) < 1e-3
    # -----------------------------------------------------
    mmd_estimator_biased = QuadraticMmdEstimator(kernel,
                                                 variance_term="sutherland_2017",
                                                 biased=True,
                                                 unit_diagonal=False)
    result_small_sample_biased = mmd_estimator_biased.forward(x=sample_x_small, y=t_xy_train[1], is_add_kernel_matrix_object=True)
    result_big_sample_biased = mmd_estimator_biased.forward(x=sample_x_big, y=t_xy_train[1], is_add_kernel_matrix_object=True)

    logger.debug(f"mmd_result_small_sample_biased: {result_small_sample_biased.mmd.item()}, mmd_result_full_sample_biased: {result_big_sample_biased.mmd.item()}")
    # -----------------------------------------------------
    # testing `from_dataset` method
    # mmd_estimator = QuadraticMmdEstimator.from_dataset(my_dataset, kernel_class=QuadraticKernelGaussianKernel)
    
    # loader = torch.utils.data.DataLoader(my_dataset, batch_size=100)
    # for x, y in loader:
    #     mmd_obj = mmd_estimator.forward(x, y)
    #     isinstance(mmd_obj, MmdValues)
    # # end for
    
    # # saving parameters and reloading estimator
    # arg_parameters = mmd_estimator.get_hyperparameters()
    # json.dumps(asdict(arg_parameters))
    # parameter_dicts = mmd_estimator.state_dict()
    
    # __kernel = QuadraticKernelGaussianKernel(**arg_parameters.kernel_object_arguments)
    # mmd_estimator = QuadraticMmdEstimator(kernel_obj=__kernel, **arg_parameters.mmd_object_arguments)
    # mmd_estimator.load_state_dict(parameter_dicts)



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
