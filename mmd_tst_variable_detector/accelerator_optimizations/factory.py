import typing as ty
from distributed import Client

from .base import BaseTaskDispatcher
from .single_cpu import SingleCpuTaskDispatcher
from .single_gpu import SingleGpuTaskDispatcher
from .dask_cpu import DaskCpuTaskDispatcher
from .concurrent_gpu import ConcurrentGpuTaskDispatcher
from ..detection_algorithm.cross_validation_detector.checkpoint_saver import (
    CheckPointSaverStabilitySelection,
)
from ..utils.post_process_logger import PostProcessLoggerHandler


def create_task_dispatcher(
    train_accelerator: str = "cpu",
    distributed_mode: str = "single",
    dask_client: ty.Optional[Client] = None,
    dask_scheduler_address: ty.Optional[str] = None,
    batch_size: int = 1,
    resume_checkpoint_saver: ty.Optional[CheckPointSaverStabilitySelection] = None,
    post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
    cv_experiment_name: ty.Optional[str] = None,
    device_id: int = 0,
    **kwargs: ty.Any,
) -> BaseTaskDispatcher:
    """Factory function creating the appropriate BaseTaskDispatcher instance.

    Parameters
    ----------
    train_accelerator : str
        Target compute hardware: 'cpu', 'gpu', or 'cuda'.
    distributed_mode : str
        Execution mode: 'single' (sequential) or 'dask' (cluster).
    dask_client : Optional[Client]
        Active Dask client if distributed_mode is 'dask'.
    dask_scheduler_address : Optional[str]
        Dask scheduler address if client is not directly provided.
    batch_size : int
        Number of tasks processed per batch.
    resume_checkpoint_saver : Optional[CheckPointSaverStabilitySelection]
        Handler to save intermediate job checkpoints.
    post_process_handler : Optional[PostProcessLoggerHandler]
        Handler to record metrics and visualizations.
    cv_experiment_name : Optional[str]
        Unique experiment identifier for logging.
    device_id : int
        Target GPU device index for single GPU execution.

    Returns
    -------
    BaseTaskDispatcher
        Concrete dispatcher configured for the requested mode.
    """
    accelerator_normalized = train_accelerator.lower()
    mode_normalized = distributed_mode.lower()

    common_kwargs = dict(
        batch_size=batch_size,
        resume_checkpoint_saver=resume_checkpoint_saver,
        post_process_handler=post_process_handler,
        cv_experiment_name=cv_experiment_name,
    )

    if mode_normalized == "single":
        if accelerator_normalized in ("gpu", "cuda"):
            return SingleGpuTaskDispatcher(device_id=device_id, **common_kwargs)
        elif accelerator_normalized == "cpu":
            return SingleCpuTaskDispatcher(**common_kwargs)
        else:
            raise ValueError(f"Unsupported train_accelerator for single mode: {train_accelerator}")

    elif mode_normalized == "dask":
        if accelerator_normalized in ("gpu", "cuda"):
            return ConcurrentGpuTaskDispatcher(
                dask_client=dask_client,
                dask_scheduler_address=dask_scheduler_address,
                **common_kwargs,
                **kwargs,
            )
        elif accelerator_normalized == "cpu":
            return DaskCpuTaskDispatcher(
                dask_client=dask_client,
                dask_scheduler_address=dask_scheduler_address,
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unsupported train_accelerator for dask mode: {train_accelerator}")

    else:
        raise ValueError(f"Unsupported distributed_mode: {distributed_mode}. Expected 'single' or 'dask'.")
