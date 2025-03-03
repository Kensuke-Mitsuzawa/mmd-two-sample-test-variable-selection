import numpy as np
import torch
import torch.utils.data

from distributed import LocalCluster

import pytorch_lightning

from mmd_tst_variable_detector import (
    RegularizationParameter,
    QuadraticKernelGaussianKernel,
    QuadraticMmdEstimator,
    SimpleDataset,
    InterpretableMmdTrainParameters,
    InterpretableMmdDetector,
    detect_variables,
    # training helper
    DefaultEarlyStoppingRule,
)
from mmd_tst_variable_detector.assessment_helper.default_settings import lr_scheduler

import logzero
logger = logzero.logger


"""This example shows you how to define a MMD-estimator.
This example is only when you know an appropriate kernel function. In other words, you know ARD weights of the Gaussian kernel.
"""


def example_interpretable_mmd(max_epochs: int):
    # hyper-parameters. Regularization parameter.
    reg_parameter_l1 = 0.1
    reg_parameter_l2 = 0.1

    # Preparation

    dimension = 20
    sample_size = 200
    noise_index = [0, 1]

    array_x = np.random.normal(size=(sample_size, dimension))
    array_y = np.random.normal(size=(sample_size, dimension))
    array_y[:, noise_index] = array_y[:, noise_index] + np.random.normal(loc=3, size=(sample_size, len(noise_index)))

    array_x_dev = np.random.normal(size=(sample_size, dimension))
    array_y_dev = np.random.normal(size=(sample_size, dimension))
    array_y_dev[:, noise_index] = array_y_dev[:, noise_index] + np.random.normal(loc=3, size=(sample_size, len(noise_index)))
    
    # Step-1: defining a dataset

    dataset_train = SimpleDataset(torch.tensor(array_x), torch.tensor(array_y))
    dataset_dev = SimpleDataset(torch.tensor(array_x_dev), torch.tensor(array_y_dev))    

    # Step-2: defining a kernel function

    # comment: In this example, I initialize ARD weights with all 1.0. However, this proper weights are supposed to be given.
    ard_weights = torch.ones(dimension)

    kernel_function = QuadraticKernelGaussianKernel(ard_weights=ard_weights)  # setting a kernel function.
    kernel_function.compute_length_scale_dataset(dataset_train)  # setting the length scale parameter of a kernel function.

    # Step-3: defining a MMD-estimator

    mmd_estimator = QuadraticMmdEstimator(kernel_obj=kernel_function)

    pl_trainer = pytorch_lightning.Trainer(max_epochs=max_epochs,
                                           callbacks=DefaultEarlyStoppingRule,
                                           enable_checkpointing=False,
                                           enable_model_summary=False,
                                           enable_progress_bar=True)
    
    training_parameter = InterpretableMmdTrainParameters(
        regularization_parameter=RegularizationParameter(reg_parameter_l1, reg_parameter_l2),
        batch_size=100,
        lr_scheduler=lr_scheduler,
        optimizer_args={"lr": 0.01},
    )

    detector = InterpretableMmdDetector(
        mmd_estimator=mmd_estimator,
        training_parameter=training_parameter,
        dataset_train=dataset_train,
        dataset_validation=dataset_dev)

    pl_trainer.fit(detector)
    
    detection_result = detector.get_trained_variables()
    variables = detect_variables(detection_result.ard_weights_kernel_k)
    
    logger.info(f"Detected variables: {variables}")
    logger.info(f"Answer of the noised variables: [0, 1]")


def test_example():
    example_interpretable_mmd(max_epochs=2000)


if __name__ == "__main__":
    test_example()
