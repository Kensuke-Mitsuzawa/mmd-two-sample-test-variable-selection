import typing as ty
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import torch
import numpy as np

from mmd_tst_variable_detector.detection_algorithm.base import BaseVariableDetector
from mmd_tst_variable_detector.detection_algorithm.path_aggregation_executor import (
    PathAggregationVariableDetector,
    PathAggregationExecutor,
    PathAggregationAlgorithmParameter,
    PathAggregationTaskRequest,
    PathAggregationTaskResult,
    PathAggregationDetectionResult,
    PathAggregator,
    transform_score_identity,
    transform_score_bounded,
    execute_optimization_task_path_aggregation,
)
from mmd_tst_variable_detector.datasets import SimpleDataset
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector import (
    InterpretableMmdTrainParameters,
    PytorchLightningDefaultArguments,
)
from tests import data_generator


def test_inheritance_and_alias():
    """Verify PathAggregationVariableDetector inherits BaseVariableDetector and alias exists."""
    assert issubclass(PathAggregationVariableDetector, BaseVariableDetector)
    assert PathAggregationExecutor is PathAggregationVariableDetector
# end def


def test_algorithm_parameters_validation():
    """Verify validation constraints in PathAggregationAlgorithmParameter."""
    params = PathAggregationAlgorithmParameter(
        regularization_grid=[0.5, 0.01, 0.1],
        threshold=0.05,
        weight_transformation="bounded",
    )
    assert params.threshold == 0.05
    assert params.weight_transformation == "bounded"
    assert params.subsampling_splits == 1

    with pytest.raises(Exception):
        # threshold must be > 0.0
        PathAggregationAlgorithmParameter(
            regularization_grid=[0.1],
            threshold=-0.5,
        )
    # end with
# end def


def test_score_transformations():
    """Verify mathematical transformations rho(t)."""
    weights = torch.tensor([0.0, 1.0, 3.0])
    
    # Identity: rho(t) = t
    out_identity = transform_score_identity(weights)
    assert torch.equal(out_identity, weights)

    # Bounded: rho(t) = t / (1 + t)
    out_bounded = transform_score_bounded(weights)
    assert torch.allclose(out_bounded, torch.tensor([0.0, 0.5, 0.75]))
# end def


def test_aggregator_logic():
    """Verify path aggregation math and variable selection."""
    params = PathAggregationAlgorithmParameter(
        regularization_grid=[0.1, 1.0],
        threshold=0.2,
        weight_transformation="identity",
        path_weights=[0.5, 0.5],
    )
    aggregator = PathAggregator(parameters=params)

    # Two tasks for two lambdas
    dummy_results = [
        PathAggregationTaskResult(
            task_id="t1",
            lambda_val=0.1,
            split_index=0,
            is_success=True,
            trained_ard_weights=[0.6, 0.1],
        ),
        PathAggregationTaskResult(
            task_id="t2",
            lambda_val=1.0,
            split_index=0,
            is_success=True,
            trained_ard_weights=[0.2, 0.0],
        ),
    ]

    res = aggregator.aggregate_path_weights(task_results=dummy_results, dimension_size=2)
    # Average score for dim 0: 0.5 * 0.6 + 0.5 * 0.2 = 0.4
    # Average score for dim 1: 0.5 * 0.1 + 0.5 * 0.0 = 0.05
    assert np.isclose(res.aggregated_scores[0], 0.4)
    assert np.isclose(res.aggregated_scores[1], 0.05)

    # Selection with tau = 0.2: only dim 0 should be selected
    selected = aggregator.select_variables(res.aggregated_scores)
    assert selected == [0]
# end def


def test_worker_optimization_routine_execution():
    """Verify execute_optimization_task_path_aggregation on small dataset."""
    torch.cuda.is_available = lambda: False
    (t_x, t_y), _ = data_generator.test_data_xy_linear(
        dim_size=4, sample_size=30, ratio_dependent_variables=0.25, random_seed=42
    )
    dataset = SimpleDataset(t_x, t_y)

    initial_ard = torch.ones(dataset.get_dimension_flattened())
    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(dataset, batch_size=len(dataset))
    kernel.set_length_scale()
    estimator = QuadraticMmdEstimator(kernel)

    base_params = InterpretableMmdTrainParameters(is_use_log=1)
    pl_config = PytorchLightningDefaultArguments(max_epochs=2, accelerator="cpu")

    request = PathAggregationTaskRequest(
        task_id="test_worker_1",
        lambda_val=0.05,
        split_index=0,
        estimator=estimator,
        dataset_train=dataset,
        dataset_validation=None,
        training_parameter=base_params,
        pytorch_trainer_config=pl_config,
    )

    result = execute_optimization_task_path_aggregation(request)
    assert isinstance(result, PathAggregationTaskResult)
    assert result.is_success is True
    assert len(result.trained_ard_weights) == 4
    assert all(w >= 0.0 for w in result.trained_ard_weights)
# end def


def test_end_to_end_single_cpu_detection():
    """Verify full PathAggregationVariableDetector pipeline on single CPU."""
    torch.cuda.is_available = lambda: False
    (t_x, t_y), ground_truth = data_generator.test_data_xy_linear(
        dim_size=6, sample_size=40, ratio_dependent_variables=0.33, random_seed=123
    )
    dataset = SimpleDataset(t_x, t_y)

    initial_ard = torch.ones(dataset.get_dimension_flattened())
    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(dataset, batch_size=len(dataset))
    kernel.set_length_scale()
    estimator = QuadraticMmdEstimator(kernel)

    base_params = InterpretableMmdTrainParameters(is_use_log=1)
    pl_config = PytorchLightningDefaultArguments(max_epochs=2, accelerator="cpu")

    detector = PathAggregationVariableDetector(
        estimator=estimator,
        regularization_grid=[0.01, 0.1],
        pytorch_trainer_config=pl_config,
        base_training_parameter=base_params,
        threshold=0.01,
        weight_transformation="identity",
        train_accelerator="cpu",
        distributed_mode="single",
    )

    result = detector.run_detection(training_dataset=dataset)

    assert isinstance(result, PathAggregationDetectionResult)
    assert len(result.aggregated_scores) == 6
    assert len(result.regularization_path_weights) == 2
    assert len(result.regularization_grid) == 2
    assert "execution_time_total_seconds" in result.execution_metadata
    assert result.execution_metadata["num_tasks_total"] == 2
# end def


def test_end_to_end_subsampling_detection():
    """Verify PathAggregationVariableDetector with subsampling splits B=2."""
    torch.cuda.is_available = lambda: False
    (t_x, t_y), _ = data_generator.test_data_xy_linear(
        dim_size=5, sample_size=50, ratio_dependent_variables=0.2, random_seed=999
    )
    dataset = SimpleDataset(t_x, t_y)

    initial_ard = torch.ones(dataset.get_dimension_flattened())
    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(dataset, batch_size=len(dataset))
    kernel.set_length_scale()
    estimator = QuadraticMmdEstimator(kernel)

    base_params = InterpretableMmdTrainParameters(is_use_log=1)
    pl_config = PytorchLightningDefaultArguments(max_epochs=2, accelerator="cpu")

    detector = PathAggregationVariableDetector(
        estimator=estimator,
        regularization_grid=[0.05, 0.2],
        pytorch_trainer_config=pl_config,
        base_training_parameter=base_params,
        threshold=0.02,
        subsampling_splits=2,
        subsampling_ratio=0.8,
        weight_transformation="bounded",
        train_accelerator="cpu",
        distributed_mode="single",
    )

    result = detector.run_detection(training_dataset=dataset)

    assert isinstance(result, PathAggregationDetectionResult)
    assert len(result.aggregated_scores) == 5
    # Total tasks executed should be splits (2) x lambda_grid (2) = 4
    assert result.execution_metadata["num_tasks_total"] == 4
    assert result.execution_metadata["weight_transformation"] == "bounded"
# end def


def test_end_to_end_dask_cpu_detection():
    """Verify PathAggregationVariableDetector with Dask CPU concurrent execution."""
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True)
    client = Client(cluster)
    try:
        (t_x, t_y), _ = data_generator.test_data_xy_linear(
            dim_size=4, sample_size=30, ratio_dependent_variables=0.25, random_seed=777
        )
        dataset = SimpleDataset(t_x, t_y)

        initial_ard = torch.ones(dataset.get_dimension_flattened())
        kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
        kernel.compute_length_scale_dataset(dataset, batch_size=len(dataset))
        kernel.set_length_scale()
        estimator = QuadraticMmdEstimator(kernel)

        base_params = InterpretableMmdTrainParameters(is_use_log=1)
        pl_config = PytorchLightningDefaultArguments(max_epochs=2, accelerator="cpu")

        detector = PathAggregationVariableDetector(
            estimator=estimator,
            regularization_grid=[0.01, 0.1],
            pytorch_trainer_config=pl_config,
            base_training_parameter=base_params,
            threshold=0.01,
            train_accelerator="cpu",
            distributed_mode="dask",
            dask_client=client,
        )

        result = detector.run_detection(training_dataset=dataset)
        assert isinstance(result, PathAggregationDetectionResult)
        assert len(result.aggregated_scores) == 4
        assert result.execution_metadata["distributed_mode"] == "dask"
    finally:
        client.close()
        cluster.close()
    # end try
# end def

