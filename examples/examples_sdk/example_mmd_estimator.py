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

    array_x = np.random.normal(size=(sample_size, dimension))
    array_y = np.random.normal(size=(sample_size, dimension))

    # Step-1: defining a dataset

    dataset_x = SimpleDataset(torch.tensor(array_x), torch.tensor(array_y))

    # Step-2: defining a kernel function

    # comment: In this example, I initialize ARD weights with all 1.0. However, this proper weights are supposed to be given.
    ard_weights = torch.ones(dimension)

    kernel_function = QuadraticKernelGaussianKernel(ard_weights=ard_weights)

    # Step-3: defining a MMD-estimator

    mmd_estimator = QuadraticMmdEstimator(kernel_obj=kernel_function)

    # Step-4: calculating MMD

    data_loader = torch.utils.data.DataLoader(dataset_x, batch_size=10, shuffle=True)

    for __pair_xy in data_loader:
        __mmd_container = mmd_estimator.forward(__pair_xy[0], __pair_xy[1])
        logger.info(f'MMD^2={__mmd_container.mmd}')
        

def test_example():
    example_mmd_estimator()
