"""Legacy compatibility shim for path_aggregation_executor.worker."""
from ..reg_aggregation_executor.worker import (
    execute_optimization_task_reg_aggregation,
    worker_reg_aggregation_optimization_routine,
    execute_optimization_task_path_aggregation,
    worker_path_aggregation_optimization_routine,
)

__all__ = [
    "execute_optimization_task_path_aggregation",
    "worker_path_aggregation_optimization_routine",
    "execute_optimization_task_reg_aggregation",
    "worker_reg_aggregation_optimization_routine",
]
