import logging
import typing as ty

from .base import BaseTaskDispatcher
from .worker import worker_execution_routine
from ..detection_algorithm.cross_validation_detector.commons import (
    RequestDistributedFunction,
    SubLearnerTrainingResult,
)
from ..logger_unit import handler

logger = logging.getLogger(f"{__package__}.{__name__}")
logger.addHandler(handler)


class SingleCpuTaskDispatcher(BaseTaskDispatcher):
    """Task dispatcher for single CPU execution.

    Executes optimization tasks sequentially in the current process on CPU.
    """

    def _execute_batch(
        self, batch: ty.List[RequestDistributedFunction]
    ) -> ty.List[SubLearnerTrainingResult]:
        batch_results: ty.List[SubLearnerTrainingResult] = []
        for task in batch:
            batch_results.append(worker_execution_routine(task))
        return batch_results
