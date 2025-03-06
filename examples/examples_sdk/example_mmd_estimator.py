import numpy as np
import torch
import torch.utils.data

from mmd_tst_variable_detector import (
    QuadraticKernelGaussianKernel,
    QuadraticMmdEstimator,
    SimpleDataset
)

import logzero
logger = logzero.logger


"""This example shows you how to define a MMD-estimator.
This example is only when you know an appropriate kernel function. In other words, you know ARD weights of the Gaussian kernel.
"""


def example_mmd_estimator():

    # Preparation

    dimension = 20
    sample_size = 200

    # target data generation
    random_generator_train = np.random.RandomState(42)
    array_x = random_generator_train.normal(size=(sample_size, dimension))
    array_y = random_generator_train.normal(size=(sample_size, dimension))
    dataset_target = SimpleDataset(torch.tensor(array_x), torch.tensor(array_y))

    # calibratiin data generation
    random_generator_dev = np.random.RandomState(24)
    array_x_dev = random_generator_dev.normal(size=(sample_size, dimension))
    array_y_dev = random_generator_dev.normal(size=(sample_size, dimension))
    dataset_calibration = SimpleDataset(torch.tensor(array_x_dev), torch.tensor(array_y_dev))

    # defining a kernel function
    # comment: In this example, I initialize ARD weights with all 1.0. However, this proper weights are supposed to be given.
    ard_weights = torch.ones(dimension)
    kernel_function = QuadraticKernelGaussianKernel(ard_weights=ard_weights)
    # I set the length scale of the kernel function with the calibration dataset.
    kernel_function.compute_length_scale_dataset(dataset_calibration)

    # defining a MMD-estimator
    mmd_estimator = QuadraticMmdEstimator(kernel_obj=kernel_function)

    # calculating MMD
    data_loader = torch.utils.data.DataLoader(dataset_target, batch_size=100, shuffle=True)
    for __pair_xy in data_loader:
        __mmd_container = mmd_estimator.forward(__pair_xy[0], __pair_xy[1])
        logger.info(f'MMD^2={__mmd_container.mmd}')
        

def test_example():
    example_mmd_estimator()


if __name__ == '__main__':
    example_mmd_estimator()
