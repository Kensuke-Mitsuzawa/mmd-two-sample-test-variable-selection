from dataclasses import dataclass
import typing as ty


@dataclass
class DistributedConfigArgs:
    """
    Args
    --------
    dask_scheduler_host: ty.Optional[str]
        Host name of dask scheduler.
        If you use local dask cluster, you can set it to 'localhost'.
    dask_scheduler_port: ty.Optional[int]
        Port number of dask scheduler.
        If you use local dask cluster, you can set it to 8786.
    is_use_local_dask_cluster: bool
        Whether you use local dask cluster or not.
        If you use local dask cluster, you can set it to True.
    n_workers: int
        Number of workers.
        This parameter is used only when you use local dask cluster.
    threads_per_worker: int
        Number of threads per worker.
        This parameter is used only when you use local dask cluster.
    distributed_mode: str
        Distributed mode.
        Either of following choices,
        1. 'single': single machine.
        2. 'dask': dask distributed.        
    """
    distributed_mode: str = 'dask'
    
    dask_scheduler_host: ty.Optional[str] = '0.0.0.0'
    dask_scheduler_port: ty.Optional[int] = 8786
    dask_dashboard_address: ty.Optional[str] = ':8787'
    dask_n_workers: int = 4
    dask_threads_per_worker: int = 4
    
    is_use_local_dask_cluster: bool = True
    
    def __post_init__(self):
        assert self.distributed_mode in ['single', 'dask', 'joblib'], f'{self.distributed_mode} is not supported.'
