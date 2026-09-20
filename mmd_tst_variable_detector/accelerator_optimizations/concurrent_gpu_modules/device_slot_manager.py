import logging
import typing as ty
from distributed import Client, Nanny, SpecCluster

logger = logging.getLogger(__name__)


class DeviceSlotManager(object):
    """Orchestrates GPU-pinned Dask worker processes using SpecCluster.

    Spawns K worker subprocesses per GPU, pinning each worker to its designated
    GPU device via the CUDA_VISIBLE_DEVICES environment variable.
    """

    @staticmethod
    def build_worker_specs(
        n_gpus: int,
        k_slots_per_gpu: int,
        memory_limit: ty.Optional[str] = None,
    ) -> ty.Dict[str, ty.Dict[str, ty.Any]]:
        """Construct the Dask worker specifications dictionary for SpecCluster.

        Parameters
        ----------
        n_gpus : int
            Number of available GPUs.
        k_slots_per_gpu : int
            Number of concurrent worker slots per GPU.
        memory_limit : Optional[str]
            Memory limit per worker (e.g. '4GB').

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Worker specification dictionary mapping worker_name to Nanny configuration.
        """
        worker_specs = {}

        for gpu_id in range(n_gpus):
            for slot_id in range(k_slots_per_gpu):
                worker_name = f"gpu_{gpu_id}_slot_{slot_id}"
                options: ty.Dict[str, ty.Any] = {
                    "name": worker_name,
                    "nthreads": 1,
                    "env": {
                        "CUDA_VISIBLE_DEVICES": str(gpu_id),
                    },
                }
                if memory_limit is not None:
                    options["memory_limit"] = memory_limit

                worker_specs[worker_name] = {
                    "cls": Nanny,
                    "options": options,
                }

        return worker_specs

    @classmethod
    def create_gpu_cluster(
        cls,
        n_gpus: int = 1,
        k_slots_per_gpu: int = 2,
        memory_limit: ty.Optional[str] = None,
        **cluster_kwargs: ty.Any,
    ) -> ty.Tuple[SpecCluster, Client]:
        """Launch a Dask SpecCluster with workers pinned to specific GPUs.

        Parameters
        ----------
        n_gpus : int
            Number of GPUs to distribute across.
        k_slots_per_gpu : int
            Number of concurrent worker slots per GPU.
        memory_limit : Optional[str]
            Optional RAM limit per worker process.
        cluster_kwargs : Any
            Additional keyword arguments for SpecCluster.

        Returns
        -------
        Tuple[SpecCluster, Client]
            Active SpecCluster and connected Dask Client.
        """
        worker_specs = cls.build_worker_specs(
            n_gpus=n_gpus,
            k_slots_per_gpu=k_slots_per_gpu,
            memory_limit=memory_limit,
        )
        logger.info(
            f"Launching SpecCluster with {len(worker_specs)} workers across {n_gpus} GPU(s) "
            f"({k_slots_per_gpu} slots/GPU)..."
        )
        cluster = SpecCluster(workers=worker_specs, **cluster_kwargs)
        client = Client(cluster)
        logger.info(f"SpecCluster initialized. Client address: {client.scheduler.address}")
        return cluster, client

    @staticmethod
    def close_cluster(
        client: ty.Optional[Client] = None,
        cluster: ty.Optional[SpecCluster] = None,
    ) -> None:
        """Gracefully shut down the client and cluster."""
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                logger.warning(f"Error closing Dask Client: {exc}")

        if cluster is not None:
            try:
                cluster.close()
            except Exception as exc:
                logger.warning(f"Error closing SpecCluster: {exc}")
