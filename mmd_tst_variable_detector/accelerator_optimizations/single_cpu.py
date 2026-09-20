import logging
import typing as ty

from .base import BaseTaskDispatcher
from ..logger_unit import handler

logger = logging.getLogger(f"{__package__}.{__name__}")
logger.addHandler(handler)


class SingleCpuTaskDispatcher(BaseTaskDispatcher):
    """Task dispatcher for single CPU execution.

    Executes optimization tasks sequentially in the current process on CPU.
    """

    def _execute_batch(
        self, batch: ty.List[ty.Any]
    ) -> ty.List[ty.Any]:
        batch_results: ty.List[ty.Any] = []
        for task in batch:
            batch_results.append(self.worker_fn(task))
        # end for
        return batch_results


