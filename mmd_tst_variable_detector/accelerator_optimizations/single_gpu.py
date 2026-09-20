import gc
import logging
import typing as ty
import torch

from .base import BaseTaskDispatcher
from .worker import worker_execution_routine
from .concurrent_gpu_modules.gpu_environment_manager import assert_device_compatibility
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


class SingleGpuTaskDispatcher(BaseTaskDispatcher):
    """Task dispatcher for single GPU execution.

    Executes optimization tasks sequentially on the target GPU,
    performing CUDA cache cleanup and garbage collection after each task.
    """

    def __init__(
        self,
        device_id: int = 0,
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
        self.device_id = device_id

    def _clean_cuda_cache(self) -> None:
        """Purge GPU memory cache and trigger garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def dispatch(
        self, seq_task_arguments: ty.List[RequestDistributedFunction]
    ) -> ty.List[SubLearnerTrainingResult]:
        """Dispatch tasks after asserting GPU device compatibility."""
        if seq_task_arguments:
            assert_device_compatibility(self.device_id)
        return super().dispatch(seq_task_arguments)

    def _execute_batch(
        self, batch: ty.List[RequestDistributedFunction]
    ) -> ty.List[SubLearnerTrainingResult]:
        batch_results: ty.List[SubLearnerTrainingResult] = []
        for task in batch:
            try:
                res = worker_execution_routine(task)
                batch_results.append(res)
            finally:
                self._clean_cuda_cache()
        return batch_results
