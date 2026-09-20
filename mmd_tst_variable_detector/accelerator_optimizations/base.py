from abc import ABC, abstractmethod
import logging
import typing as ty

from ..logger_unit import handler
from ..detection_algorithm.cross_validation_detector.commons import (
    RequestDistributedFunction,
    SubLearnerTrainingResult,
)
from ..detection_algorithm.cross_validation_detector.checkpoint_saver import (
    CheckPointSaverStabilitySelection,
)
from ..utils.post_process_logger import PostProcessLoggerHandler

logger = logging.getLogger(f"{__package__}.{__name__}")
logger.addHandler(handler)


class BaseTaskDispatcher(ABC):
    """Abstract base class for all task dispatchers.

    Provides common batch orchestration, checkpoint saving, and post-process logging hooks.
    Specialized subclasses only need to implement `_execute_batch`.
    """

    def __init__(
        self,
        batch_size: int = 1,
        resume_checkpoint_saver: ty.Optional[CheckPointSaverStabilitySelection] = None,
        post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
        cv_experiment_name: ty.Optional[str] = None,
    ) -> None:
        self.batch_size = max(1, batch_size)
        self.resume_checkpoint_saver = resume_checkpoint_saver
        self.post_process_handler = post_process_handler
        self.cv_experiment_name = cv_experiment_name

    def dispatch(
        self, seq_task_arguments: ty.List[RequestDistributedFunction]
    ) -> ty.List[SubLearnerTrainingResult]:
        """Dispatch a list of optimization tasks in batches.

        Parameters
        ----------
        seq_task_arguments : List[RequestDistributedFunction]
            The list of task arguments to execute.

        Returns
        -------
        List[SubLearnerTrainingResult]
            Aggregated results from all executed tasks.
        """
        if not seq_task_arguments:
            return []

        seq_batch = [
            seq_task_arguments[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range((len(seq_task_arguments) + self.batch_size - 1) // self.batch_size)
        ]

        seq_results: ty.List[SubLearnerTrainingResult] = []
        for on_job_batch in seq_batch:
            batch_results = self._execute_batch(on_job_batch)
            seq_results.extend(batch_results)

            if self.resume_checkpoint_saver is not None:
                for sub_learner_result in batch_results:
                    if isinstance(sub_learner_result, SubLearnerTrainingResult):
                        self.resume_checkpoint_saver.save_checkpoint(sub_learner_result)

            if self.post_process_handler is not None:
                self._log_post_process(batch_results)

        return seq_results

    @abstractmethod
    def _execute_batch(
        self, batch: ty.List[RequestDistributedFunction]
    ) -> ty.List[SubLearnerTrainingResult]:
        """Execute a single batch of tasks. Must be implemented by subclasses."""
        raise NotImplementedError

    def _log_post_process(
        self, seq_results_one_batch: ty.List[SubLearnerTrainingResult]
    ) -> None:
        """Log post-processing metrics and artifacts for completed batch results."""
        if self.post_process_handler is None:
            return
        logger.debug("Logging post-process results...")
        for res in seq_results_one_batch:
            run_name = res.get_job_id_string()
            loggers = self.post_process_handler.initialize_logger(
                run_name=run_name, group_name=self.cv_experiment_name
            )
            self.post_process_handler.log(loggers=loggers, target_object=res)
        logger.debug("Logging Done")
