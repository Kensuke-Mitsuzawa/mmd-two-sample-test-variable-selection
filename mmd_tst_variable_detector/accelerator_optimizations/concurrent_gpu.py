import gc
import logging
import typing as ty
from distributed import Client
import torch

from .base import BaseTaskDispatcher
from .concurrent_gpu_modules.gpu_environment_manager import (

    GpuEnvironmentManager,
    assert_device_compatibility,
)
from .concurrent_gpu_modules.vram_estimator import VramConsumptionEstimator
from .concurrent_gpu_modules.device_slot_manager import DeviceSlotManager
from ..utils.post_process_logger import PostProcessLoggerHandler
from ..logger_unit import handler


logger = logging.getLogger(f"{__package__}.{__name__}")
logger.addHandler(handler)


class ConcurrentGpuTaskDispatcher(BaseTaskDispatcher):
    """Task dispatcher for concurrent GPU execution across GPU-pinned Dask workers.

    Coordinates multi-slot GPU execution with NVIDIA MPS, VRAM estimation, and
    per-worker GPU pinning via Dask SpecCluster.
    """

    def __init__(
        self,
        dask_client: ty.Optional[Client] = None,
        dask_scheduler_address: ty.Optional[str] = None,
        batch_size: int = 1,
        resume_checkpoint_saver: ty.Optional[ty.Any] = None,
        post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
        cv_experiment_name: ty.Optional[str] = None,
        n_gpus: ty.Optional[int] = None,
        k_slots_per_gpu: ty.Optional[int] = None,
        enable_mps: bool = True,
        vram_safety_margin: float = 0.85,
        memory_limit: ty.Optional[str] = None,
        slot_manager: ty.Optional[ty.Any] = None,
        worker_fn: ty.Optional[ty.Callable] = None,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            resume_checkpoint_saver=resume_checkpoint_saver,
            post_process_handler=post_process_handler,
            cv_experiment_name=cv_experiment_name,
            worker_fn=worker_fn,
        )
        self.dask_client = dask_client
        self.dask_scheduler_address = dask_scheduler_address
        self.n_gpus = n_gpus
        self.k_slots_per_gpu = k_slots_per_gpu
        self.enable_mps = enable_mps
        self.vram_safety_margin = vram_safety_margin
        self.memory_limit = memory_limit
        self.slot_manager = slot_manager

        self._managed_cluster = None
        self._managed_client = None
        self._mps_manager = None

    def _prepare_cluster(
        self, sample_task: ty.Optional[ty.Any] = None
    ) -> Client:
        """Ensure a Dask client and cluster are ready for execution."""
        if self.dask_client is not None:
            return self.dask_client

        if self.dask_scheduler_address is not None:
            self._managed_client = Client(self.dask_scheduler_address)
            return self._managed_client

        # Initialize MPS if requested
        if self.enable_mps:
            self._mps_manager = GpuEnvironmentManager(enable_mps=True)
            self._mps_manager.start_mps()

        n_gpus = self.n_gpus if self.n_gpus is not None else max(1, GpuEnvironmentManager.get_gpu_count())

        # Determine K slots per GPU
        if self.k_slots_per_gpu is not None:
            k_slots = self.k_slots_per_gpu
        else:
            estimator = VramConsumptionEstimator(
                device_id=0,
                safety_margin=self.vram_safety_margin,
            )
            k_slots = estimator.estimate_from_task(sample_task)

        cluster, client = DeviceSlotManager.create_gpu_cluster(
            n_gpus=n_gpus,
            k_slots_per_gpu=k_slots,
            memory_limit=self.memory_limit,
        )
        self._managed_cluster = cluster
        self._managed_client = client
        return client

    def _teardown_managed_cluster(self) -> None:
        """Tear down self-managed cluster and MPS daemon."""
        if self._managed_client is not None or self._managed_cluster is not None:
            DeviceSlotManager.close_cluster(
                client=self._managed_client,
                cluster=self._managed_cluster,
            )
            self._managed_client = None
            self._managed_cluster = None

        if self._mps_manager is not None:
            self._mps_manager.stop_mps()
            self._mps_manager = None

    def dispatch(
        self, seq_task_arguments: ty.List[ty.Any]
    ) -> ty.List[ty.Any]:
        """Dispatch optimization tasks across concurrent GPU workers.

        Enforces device compatibility check, prepares cluster, and tears down
        self-managed cluster upon completion.
        """
        if not seq_task_arguments:
            return []
        # end if

        # Enforce device compatibility check on all target GPUs before execution
        target_gpus = self.n_gpus if self.n_gpus is not None else max(1, GpuEnvironmentManager.get_gpu_count())
        for dev_id in range(target_gpus):
            assert_device_compatibility(dev_id)
        # end for

        try:
            return super().dispatch(seq_task_arguments)
        finally:
            self._teardown_managed_cluster()
        # end try

    def _execute_batch(
        self, batch: ty.List[ty.Any]
    ) -> ty.List[ty.Any]:
        """Execute a batch of tasks concurrently on the Dask GPU cluster."""
        client = self._prepare_cluster(sample_task=batch[0] if batch else None)

        # Strip overhead from PyTorch Lightning trainer configuration
        for task in batch:
            trainer = getattr(task, "trainer_lightning", None)
            if trainer is not None:
                trainer.enable_progress_bar = False
                trainer.enable_model_summary = False
                trainer.enable_checkpointing = False
            # end if
        # end for

        task_queue = client.map(self.worker_fn, batch)
        batch_results: ty.List[ty.Any] = client.gather(task_queue)  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        # end if

        return batch_results


