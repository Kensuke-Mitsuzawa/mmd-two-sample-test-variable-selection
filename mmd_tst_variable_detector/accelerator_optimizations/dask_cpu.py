import logging
import typing as ty
from distributed import Client

from .base import BaseTaskDispatcher
from .worker import worker_execution_routine
from ..detection_algorithm.cross_validation_detector.commons import (
    RequestDistributedFunction,
    SubLearnerTrainingResult,
)
from ..detection_algorithm.cross_validation_detector.checkpoint_saver import (
    CheckPointSaverStabilitySelection,
)
from ..utils.post_process_logger import PostProcessLoggerHandler
from ..logger_unit import handler

logger = logging.getLogger(f"{__package__}.{__name__}")
logger.addHandler(handler)


class DaskCpuTaskDispatcher(BaseTaskDispatcher):
    """Task dispatcher for distributed CPU execution via Dask.

    Maps optimization tasks across CPU workers using a Dask Client.
    """

    def __init__(
        self,
        dask_client: ty.Optional[Client] = None,
        dask_scheduler_address: ty.Optional[str] = None,
        batch_size: int = 1,
        resume_checkpoint_saver: ty.Optional[CheckPointSaverStabilitySelection] = None,
        post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
        cv_experiment_name: ty.Optional[str] = None,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            resume_checkpoint_saver=resume_checkpoint_saver,
            post_process_handler=post_process_handler,
            cv_experiment_name=cv_experiment_name,
        )
        self._dask_client = dask_client
        self._dask_scheduler_address = dask_scheduler_address

    def _get_client(self) -> Client:
        if self._dask_client is not None:
            return self._dask_client
        if self._dask_scheduler_address is not None:
            return Client(self._dask_scheduler_address)
        # Fall back to default/current client
        return Client.current()

    def _execute_batch(
        self, batch: ty.List[RequestDistributedFunction]
    ) -> ty.List[SubLearnerTrainingResult]:
        client = self._get_client()
        task_queue = client.map(worker_execution_routine, batch)
        batch_results: ty.List[SubLearnerTrainingResult] = client.gather(task_queue)  # type: ignore
        return batch_results
