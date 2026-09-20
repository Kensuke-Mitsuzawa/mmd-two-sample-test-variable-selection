from .base import BaseTaskDispatcher
from .single_cpu import SingleCpuTaskDispatcher
from .single_gpu import SingleGpuTaskDispatcher
from .dask_cpu import DaskCpuTaskDispatcher
from .concurrent_gpu import ConcurrentGpuTaskDispatcher
from .factory import create_task_dispatcher
from .concurrent_gpu_modules import (
    IncompatibleGpuArchitectureError,
    GpuEnvironmentManager,
    assert_device_compatibility,
    VramConsumptionEstimator,
    DeviceSlotManager,
)

__all__ = [
    "BaseTaskDispatcher",
    "SingleCpuTaskDispatcher",
    "SingleGpuTaskDispatcher",
    "DaskCpuTaskDispatcher",
    "ConcurrentGpuTaskDispatcher",
    "create_task_dispatcher",
    "IncompatibleGpuArchitectureError",
    "GpuEnvironmentManager",
    "assert_device_compatibility",
    "VramConsumptionEstimator",
    "DeviceSlotManager",
]


def __getattr__(name: str):
    if name == "worker_execution_routine":
        from .worker import worker_execution_routine
        return worker_execution_routine
    # end if
    raise AttributeError(f"module {__name__} has no attribute {name}")

