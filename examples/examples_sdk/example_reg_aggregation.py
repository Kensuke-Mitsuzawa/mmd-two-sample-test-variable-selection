import typing as ty
import numpy as np
import torch
import torch.utils.data

from mmd_tst_variable_detector import (
    QuadraticKernelGaussianKernel,
    QuadraticMmdEstimator,
    SimpleDataset,
    InterpretableMmdTrainParameters,
    PytorchLightningDefaultArguments,
    RegAggregationVariableDetector,
    RegAggregationDetectionResult,
    # Legacy alias for backward compatibility:
    PathAggregationVariableDetector,
)
from mmd_tst_variable_detector.detection_algorithm.early_stoppings import ConvergenceEarlyStop
from mmd_tst_variable_detector.assessment_helper.default_settings import lr_scheduler

import logzero
logger = logzero.logger


"""This example demonstrates variable detection using the Regularization Aggregation algorithm (RegAggregation).
Formerly known as Path Aggregation (path_aggregation_executor / PathAggregationVariableDetector).

The RegAggregation algorithm computes ARD weights along a grid of L1 regularization parameters
(the regularization path), aggregates their importance scores, and selects variables exceeding
a prescribed threshold or percentile cutoff.
"""


def example_reg_aggregation(max_epochs: int = 10) -> None:
    """Run example of RegAggregationVariableDetector.

    Args:
        max_epochs: Maximum epochs per regularization level optimization.
    """
    # 1. Preparation of synthetic two-sample dataset
    dimension = 20
    sample_size = 200
    noise_index = [0, 1]

    array_x = np.random.normal(size=(sample_size, dimension))
    array_y = np.random.normal(size=(sample_size, dimension))
    array_y[:, noise_index] = array_y[:, noise_index] + np.random.normal(loc=3, size=(sample_size, len(noise_index)))

    # Step-1: defining dataset
    dataset_train = SimpleDataset(torch.tensor(array_x, dtype=torch.float32), torch.tensor(array_y, dtype=torch.float32))

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
        is_use_log=1,
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

    # Step-5: instantiate RegAggregationVariableDetector
    # Notice: PathAggregationVariableDetector can be used interchangeably as a legacy alias.
    regularization_grid = [0.001, 0.01, 0.05, 0.1]

    detector = RegAggregationVariableDetector(
        estimator=mmd_estimator,
        regularization_grid=regularization_grid,
        pytorch_trainer_config=pytorch_trainer_config,
        base_training_parameter=base_training_param,
        threshold=0.01,
        weight_transformation="identity",
        train_accelerator=accelerator,
        distributed_mode="single",
    )

    # Step-6: run detection workflow
    result: RegAggregationDetectionResult = detector.run_detection(
        training_dataset=dataset_train,
    )

    logger.info(f"Selected variables: {result.selected_variables}")
    logger.info(f"Aggregated importance scores: {result.aggregated_scores}")
    logger.info(f"Regularization path weights: {result.regularization_path_weights}")
# end def


def test_example() -> None:
    example_reg_aggregation(max_epochs=5)
# end def


if __name__ == '__main__':
    example_reg_aggregation(max_epochs=10)
# end if
