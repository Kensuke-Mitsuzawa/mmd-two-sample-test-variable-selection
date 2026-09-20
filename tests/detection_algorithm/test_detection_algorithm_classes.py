import typing as ty
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch
import numpy as np

from mmd_tst_variable_detector.detection_algorithm.base import (
    BaseVariableDetector,
    BaseMmdOptimizationExecutor,
)
from mmd_tst_variable_detector.detection_algorithm.baseline_mmd import (
    BaselineMmdVariableDetector,
    BaselineMmdResult,
    baseline_mmd,
)
from mmd_tst_variable_detector.detection_algorithm.detection_algorithm_one import (
    AlgorithmOneVariableDetector,
    AlgorithmOneResult,
    detection_algorithm_one,
)
from mmd_tst_variable_detector.detection_algorithm.cross_validation_detector.cross_validation_detector import (
    CrossValidationInterpretableVariableDetector,
    CrossValidationTrainedParameter,
)
from mmd_tst_variable_detector.interface.data_objects import BasicVariableSelectionResult
from mmd_tst_variable_detector.interface.interface import Interface
from mmd_tst_variable_detector.datasets import SimpleDataset
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector import (
    InterpretableMmdTrainParameters,
    PytorchLightningDefaultArguments,
)
from tests import data_generator


def test_detector_inheritance_hierarchy():
    """Verify all 3 detector classes inherit from BaseVariableDetector and BaseMmdOptimizationExecutor alias."""
    assert BaseMmdOptimizationExecutor is BaseVariableDetector
    assert issubclass(CrossValidationInterpretableVariableDetector, BaseVariableDetector)
    assert issubclass(AlgorithmOneVariableDetector, BaseVariableDetector)
    assert issubclass(BaselineMmdVariableDetector, BaseVariableDetector)
# end def


def test_baseline_detector_class_run_detection(resource_path_root: Path):
    """Test BaselineMmdVariableDetector execution via run_detection."""
    torch.cuda.is_available = lambda: False
    t_xy_train, __ = data_generator.test_data_xy_linear(sample_size=100, random_seed=42)
    dataset_train = SimpleDataset(t_xy_train[0], t_xy_train[1])

    initial_ard = torch.ones(dataset_train.get_dimension_flattened())
    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(dataset_train, batch_size=len(dataset_train))
    kernel.set_length_scale()
    mmd_estimator = QuadraticMmdEstimator(kernel)

    base_training_parameter = InterpretableMmdTrainParameters(
        is_use_log=0,
    )
    pl_config = PytorchLightningDefaultArguments(
        max_epochs=2,
        accelerator='cpu',
    )

    detector = BaselineMmdVariableDetector(
        estimator=mmd_estimator,
        training_parameter=base_training_parameter,
        pytorch_trainer_config=pl_config,
    )

    assert isinstance(detector, BaseVariableDetector)
    result = detector.run_detection(training_dataset=dataset_train)

    assert isinstance(result, BaselineMmdResult)
    assert result.selected_variables is not None
    assert result.trained_ard_weights is not None
# end def


def test_cv_detector_run_detection_delegation():
    """Verify CrossValidationInterpretableVariableDetector.run_detection calls run_cv_detection."""
    detector = CrossValidationInterpretableVariableDetector.__new__(CrossValidationInterpretableVariableDetector)
    detector.run_cv_detection = MagicMock(return_value="mock_cv_result")

    dummy_train = MagicMock(spec=SimpleDataset)
    dummy_val = MagicMock(spec=SimpleDataset)

    res = detector.run_detection(training_dataset=dummy_train, validation_dataset=dummy_val)
    assert res == "mock_cv_result"
    detector.run_cv_detection.assert_called_once_with(
        training_dataset=dummy_train,
        validation_dataset=dummy_val,
    )
# end def


def test_algorithm_one_detector_delegation():
    """Verify AlgorithmOneVariableDetector conforms to BaseVariableDetector and is invoked by detection_algorithm_one."""
    assert issubclass(AlgorithmOneVariableDetector, BaseVariableDetector)

    dummy_estimator = MagicMock()
    dummy_train_param = MagicMock()

    detector = AlgorithmOneVariableDetector(
        estimator=dummy_estimator,
        base_training_parameter=dummy_train_param,
    )
    assert isinstance(detector, BaseVariableDetector)
    assert hasattr(detector, 'run_detection')
# end def


def test_basic_variable_selection_result_has_detector_field():
    """Verify BasicVariableSelectionResult supports detector field."""
    res = BasicVariableSelectionResult(
        weights=[1.0, 0.0],
        variables=[0],
        p_value=0.01,
        detector=MagicMock(spec=BaseVariableDetector),
    )
    assert res.detector is not None
    assert isinstance(res.detector, BaseVariableDetector)
# end def
