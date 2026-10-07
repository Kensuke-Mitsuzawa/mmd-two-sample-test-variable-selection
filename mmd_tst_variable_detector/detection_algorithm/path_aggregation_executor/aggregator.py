"""Legacy compatibility shim for path_aggregation_executor.aggregator."""
from ..reg_aggregation_executor.aggregator import (
    RegAggregator,
    AggregatedRegScores,
    transform_score_identity,
    transform_score_bounded,
    PathAggregator,
    AggregatedPathScores,
)

__all__ = [
    "PathAggregator",
    "AggregatedPathScores",
    "transform_score_identity",
    "transform_score_bounded",
    "RegAggregator",
    "AggregatedRegScores",
]
