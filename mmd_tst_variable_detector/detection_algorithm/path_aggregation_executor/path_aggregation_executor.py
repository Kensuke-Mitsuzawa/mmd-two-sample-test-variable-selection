"""Legacy compatibility shim for path_aggregation_executor.path_aggregation_executor."""
from ..reg_aggregation_executor.reg_aggregation_executor import (
    RegAggregationVariableDetector,
    RegAggregationExecutor,
    PathAggregationVariableDetector,
    PathAggregationExecutor,
)

__all__ = [
    "PathAggregationVariableDetector",
    "PathAggregationExecutor",
    "RegAggregationVariableDetector",
    "RegAggregationExecutor",
]
