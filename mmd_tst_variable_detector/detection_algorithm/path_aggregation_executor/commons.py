"""Legacy compatibility shim for path_aggregation_executor.commons."""
from ..reg_aggregation_executor.commons import (
    RegAggregationAlgorithmParameter,
    RegAggregationTaskRequest,
    RegAggregationTaskResult,
    RegAggregationDetectionResult,
    PathAggregationAlgorithmParameter,
    PathAggregationTaskRequest,
    PathAggregationTaskResult,
    PathAggregationDetectionResult,
)

__all__ = [
    "PathAggregationAlgorithmParameter",
    "PathAggregationTaskRequest",
    "PathAggregationTaskResult",
    "PathAggregationDetectionResult",
    "RegAggregationAlgorithmParameter",
    "RegAggregationTaskRequest",
    "RegAggregationTaskResult",
    "RegAggregationDetectionResult",
]
