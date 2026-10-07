import typing as ty
import numpy as np
import torch
import torch.utils.data

from mmd_tst_variable_detector import (
    QuadraticKernelGaussianKernel,
    QuadraticMmdEstimator,
    SimpleDataset,
    RegularizationParameter,
    InterpretableMmdTrainParameters,
    PytorchLightningDefaultArguments,
    RegularizationSearchParameters,
    RegSelectionVariableDetector,
    RegSelectionResult,
    # Legacy alias for backward compatibility:
    AlgorithmOneVariableDetector,
)
from mmd_tst_variable_detector.detection_algorithm.early_stoppings import ConvergenceEarlyStop
from mmd_tst_variable_detector.assessment_helper.default_settings import lr_scheduler

import logzero
logger = logzero.logger


"""This example demonstrates variable detection using the Regularization Selection algorithm (RegSelection).
Formerly known as Algorithm 1 (detection_algorithm_one / AlgorithmOneVariableDetector).

The RegSelection algorithm optimizes ARD weights over a search space of regularization parameters
(either via Optuna search_objective_based or auto_min_max_range) and selects the best model.
"""


def example_reg_selection(max_epochs: int = 20) -> None:
    """Run example of RegSelectionVariableDetector.

    Args:
        max_epochs: Maximum epochs per optimization trial.
    """
    # 1. Preparation of synthetic two-sample dataset
    dimension = 20
    sample_size = 200
    noise_index = [0, 1]

    array_x = np.random.normal(size=(sample_size, dimension))
    array_y = np.random.normal(size=(sample_size, dimension))
    array_y[:, noise_index] = array_y[:, noise_index] + np.random.normal(loc=3, size=(sample_size, len(noise_index)))

    array_x_dev = np.random.normal(size=(sample_size, dimension))
    array_y_dev = np.random.normal(size=(sample_size, dimension))
    array_y_dev[:, noise_index] = array_y_dev[:, noise_index] + np.random.normal(loc=3, size=(sample_size, len(noise_index)))

    # Step-1: defining datasets
    dataset_train = SimpleDataset(torch.tensor(array_x, dtype=torch.float32), torch.tensor(array_y, dtype=torch.float32))
    dataset_dev = SimpleDataset(torch.tensor(array_x_dev, dtype=torch.float32), torch.tensor(array_y_dev, dtype=torch.float32))

    # Step-2: defining a kernel function and length scale
    ard_weights = torch.ones(dimension)
    kernel_function = QuadraticKernelGaussianKernel(ard_weights=ard_weights)
    kernel_function.compute_length_scale_dataset(dataset_train)
    kernel_function.set_length_scale()

    # Step-3: defining an MMD estimator
    mmd_estimator = QuadraticMmdEstimator(kernel_obj=kernel_function)

    # Step-4: configure training parameters
    base_training_param = InterpretableMmdTrainParameters(
        lr_scheduler=lr_scheduler,
        optimizer_args={'lr': 0.05},
        is_use_log=0,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    pytorch_trainer_config = PytorchLightningDefaultArguments(
        max_epochs=max_epochs,
        callbacks=ConvergenceEarlyStop(ignore_epochs=5),
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        accelerator=accelerator,
    )

    reg_search_parameters = RegularizationSearchParameters(
        n_regularization_parameter=3,
        n_search_iteration=2,
        max_concurrent_job=1,
    )

    # Step-5: instantiate RegSelectionVariableDetector
    # Notice: AlgorithmOneVariableDetector can be used interchangeably as a legacy alias.
    detector = RegSelectionVariableDetector(
        estimator=mmd_estimator,
        base_training_parameter=base_training_param,
        pytorch_trainer_config=pytorch_trainer_config,
        candidate_regularization_parameters='auto_min_max_range',
        regularization_search_parameter=reg_search_parameters,
        n_permutation_test=50,
        distributed_mode='single',
    )

    # Step-6: run detection workflow
    result: RegSelectionResult = detector.run_detection(
        training_dataset=dataset_train,
        validation_dataset=dataset_dev,
    )

    logger.info(f"Selected model regularization: {result.selected_model.regularization_parameter}")
    logger.info(f"Selected variables: {result.selected_model.selected_variables}")
    logger.info(f"Dev p-value: {result.selected_model.p_value_dev}")
# end def


def test_example() -> None:
    example_reg_selection(max_epochs=5)
# end def


if __name__ == '__main__':
    example_reg_selection(max_epochs=10)
# end if
