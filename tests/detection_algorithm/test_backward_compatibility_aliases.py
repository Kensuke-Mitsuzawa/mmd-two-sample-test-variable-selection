"""Tests verifying 100% backward compatibility of renamed modules and classes.

Validates that:
1. Legacy module paths can still be imported without error.
2. Legacy class names and function names are exact aliases (referential identity).
3. Legacy and new interfaces produce identical results.
"""
import pytest
from unittest.mock import MagicMock

from mmd_tst_variable_detector.detection_algorithm.base import BaseVariableDetector

# New modules
from mmd_tst_variable_detector.detection_algorithm.reg_selection import (
    RegSelectionVariableDetector,
    RegSelectionResult,
    RegSelectionIndividualResult,
    detection_reg_selection,
)
from mmd_tst_variable_detector.detection_algorithm.reg_aggregation_executor import (
    RegAggregationVariableDetector,
    RegAggregationExecutor,
    RegAggregationAlgorithmParameter,
    RegAggregationTaskRequest,
    RegAggregationTaskResult,
    RegAggregationDetectionResult,
    RegAggregator,
    AggregatedRegScores,
    execute_optimization_task_reg_aggregation,
    worker_reg_aggregation_optimization_routine,
)

# Legacy module shims
from mmd_tst_variable_detector.detection_algorithm.detection_algorithm_one import (
    AlgorithmOneVariableDetector,
    AlgorithmOneResult,
    AlgorithmOneIndividualResult,
    detection_algorithm_one,
)
from mmd_tst_variable_detector.detection_algorithm.path_aggregation_executor import (
    PathAggregationVariableDetector,
    PathAggregationExecutor,
    PathAggregationAlgorithmParameter,
    PathAggregationTaskRequest,
    PathAggregationTaskResult,
    PathAggregationDetectionResult,
    PathAggregator,
    AggregatedPathScores,
    execute_optimization_task_path_aggregation,
    worker_path_aggregation_optimization_routine,
)

# Interface configuration
from mmd_tst_variable_detector.interface.module_configs.algorithm_configs.algorithm_config import (
    RegSelectionConfigArgs,
    AlgorithmOneConfigArgs,
)


def test_reg_selection_legacy_aliases_identity():
    """Verify algorithm_one class and function names are exact aliases of reg_selection."""
    assert AlgorithmOneVariableDetector is RegSelectionVariableDetector
    assert AlgorithmOneResult is RegSelectionResult
    assert AlgorithmOneIndividualResult is RegSelectionIndividualResult
    assert detection_algorithm_one is detection_reg_selection
    assert AlgorithmOneConfigArgs is RegSelectionConfigArgs
# end def


def test_reg_aggregation_legacy_aliases_identity():
    """Verify path_aggregation class and function names are exact aliases of reg_aggregation."""
    assert PathAggregationVariableDetector is RegAggregationVariableDetector
    assert PathAggregationExecutor is RegAggregationExecutor
    assert PathAggregationAlgorithmParameter is RegAggregationAlgorithmParameter
    assert PathAggregationTaskRequest is RegAggregationTaskRequest
    assert PathAggregationTaskResult is RegAggregationTaskResult
    assert PathAggregationDetectionResult is RegAggregationDetectionResult
    assert PathAggregator is RegAggregator
    assert AggregatedPathScores is AggregatedRegScores
    assert execute_optimization_task_path_aggregation is execute_optimization_task_reg_aggregation
    assert worker_path_aggregation_optimization_routine is worker_reg_aggregation_optimization_routine
# end def


def test_inheritance_and_subclass_check():
    """Verify subclass and instance checks work interchangeably with aliases."""
    assert issubclass(AlgorithmOneVariableDetector, BaseVariableDetector)
    assert issubclass(RegSelectionVariableDetector, BaseVariableDetector)
    assert issubclass(PathAggregationVariableDetector, BaseVariableDetector)
    assert issubclass(RegAggregationVariableDetector, BaseVariableDetector)

    dummy_estimator = MagicMock()
    dummy_param = MagicMock()

    inst1 = AlgorithmOneVariableDetector(estimator=dummy_estimator, base_training_parameter=dummy_param)
    assert isinstance(inst1, AlgorithmOneVariableDetector)
    assert isinstance(inst1, RegSelectionVariableDetector)

    inst2 = PathAggregationVariableDetector(estimator=dummy_estimator, regularization_grid=[0.1])
    assert isinstance(inst2, PathAggregationVariableDetector)
    assert isinstance(inst2, RegAggregationVariableDetector)
# end def
