from .commons import (
    RegAggregationAlgorithmParameter,
    RegAggregationTaskRequest,
    RegAggregationTaskResult,
    RegAggregationDetectionResult,
    PathAggregationAlgorithmParameter,
    PathAggregationTaskRequest,
    PathAggregationTaskResult,
    PathAggregationDetectionResult,
)
from .worker import (
    execute_optimization_task_reg_aggregation,
    worker_reg_aggregation_optimization_routine,
    execute_optimization_task_path_aggregation,
    worker_path_aggregation_optimization_routine,
)
from .aggregator import (
    RegAggregator,
    AggregatedRegScores,
    transform_score_identity,
    transform_score_bounded,
    PathAggregator,
    AggregatedPathScores,
)
from .reg_aggregation_executor import (
    RegAggregationVariableDetector,
    RegAggregationExecutor,
    PathAggregationVariableDetector,
    PathAggregationExecutor,
)

__all__ = [
    # New names
    "RegAggregationAlgorithmParameter",
    "RegAggregationTaskRequest",
    "RegAggregationTaskResult",
    "RegAggregationDetectionResult",
    "execute_optimization_task_reg_aggregation",
    "worker_reg_aggregation_optimization_routine",
    "RegAggregator",
    "AggregatedRegScores",
    "transform_score_identity",
    "transform_score_bounded",
    "RegAggregationVariableDetector",
    "RegAggregationExecutor",
    # Legacy aliases
    "PathAggregationAlgorithmParameter",
    "PathAggregationTaskRequest",
    "PathAggregationTaskResult",
    "PathAggregationDetectionResult",
    "execute_optimization_task_path_aggregation",
    "worker_path_aggregation_optimization_routine",
    "PathAggregator",
    "AggregatedPathScores",
    "PathAggregationVariableDetector",
    "PathAggregationExecutor",
]
