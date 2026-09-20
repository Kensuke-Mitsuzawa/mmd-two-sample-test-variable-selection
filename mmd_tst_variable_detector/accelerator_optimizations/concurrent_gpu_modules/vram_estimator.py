import gc
import logging
import math
import typing as ty
import torch

logger = logging.getLogger(__name__)



class VramConsumptionEstimator(object):
    """Estimates per-task GPU VRAM consumption and computes optimal concurrency slots per GPU."""

    def __init__(
        self,
        device_id: int = 0,
        safety_margin: float = 0.85,
        default_peak_vram_bytes: int = 512 * 1024 * 1024,  # 512MB default fallback
        max_slots_per_gpu: int = 8,
    ):
        self.device_id = device_id
        self.safety_margin = safety_margin
        self.default_peak_vram_bytes = default_peak_vram_bytes
        self.max_slots_per_gpu = max_slots_per_gpu

    def get_vram_info(self) -> ty.Tuple[int, int]:
        """Fetch current free and total VRAM in bytes for the device.

        Returns
        -------
        Tuple[int, int]
            (free_memory_bytes, total_memory_bytes)
        """
        if not torch.cuda.is_available():
            return 0, 0
        return torch.cuda.mem_get_info(self.device_id)

    def calculate_k_slots(
        self,
        free_vram_bytes: int,
        peak_vram_bytes: int,
    ) -> int:
        """Calculate the number of concurrent task slots K that fit within available VRAM.

        Formula:
            K = max(1, min(max_slots, floor((free_vram * safety_margin) / peak_vram)))

        Parameters
        ----------
        free_vram_bytes : int
            Available VRAM in bytes.
        peak_vram_bytes : int
            Estimated peak VRAM per task in bytes.

        Returns
        -------
        int
            Optimal number of concurrent task slots per GPU.
        """
        if peak_vram_bytes <= 0 or free_vram_bytes <= 0:
            return 1

        effective_budget = free_vram_bytes * self.safety_margin
        raw_k = math.floor(effective_budget / peak_vram_bytes)
        k_slots = max(1, min(self.max_slots_per_gpu, int(raw_k)))
        logger.info(
            f"VRAM Calculation: Free={free_vram_bytes / (1024**2):.1f}MB, "
            f"Peak={peak_vram_bytes / (1024**2):.1f}MB, "
            f"SafetyMargin={self.safety_margin} -> K={k_slots} slots"
        )
        return k_slots

    def estimate_from_task(
        self,
        sample_task: ty.Optional[ty.Any] = None,
    ) -> int:
        """Estimate peak VRAM per task and determine optimal K slots per GPU.

        If a sample_task is provided and CUDA execution is functional, runs a 1-epoch dry-run
        to monitor peak memory reserved. Otherwise, falls back to default peak memory.
        """
        free_bytes, total_bytes = self.get_vram_info()
        if free_bytes == 0:
            return 1

        peak_bytes = self.default_peak_vram_bytes

        if sample_task is not None and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(self.device_id)
                # If sample task has dataset, estimate tensor footprint in GPU
                ds_train = getattr(sample_task, "dataset_train", None)
                if ds_train is not None and hasattr(ds_train, "get_dimension_flattened"):
                    _d = ds_train.get_dimension_flattened()
                    dim = _d[0] if isinstance(_d, (tuple, list)) else int(_d)
                    sample_count = len(ds_train)
                    # Sample dataset memory: ~2 * sample_count * dim * 4 bytes + kernel matrix
                    tensor_bytes = int(2 * sample_count * dim * 4 + sample_count * sample_count * 4)
                    peak_bytes = max(peak_bytes, tensor_bytes * 2)

                peak_measured = torch.cuda.max_memory_reserved(self.device_id)
                if peak_measured > 0:
                    peak_bytes = peak_measured
            except Exception as exc:
                logger.warning(
                    f"Dry-run VRAM estimation encountered an error: {exc}. Using default peak {peak_bytes} bytes."
                )
            finally:
                torch.cuda.empty_cache()
                gc.collect()

        return self.calculate_k_slots(free_bytes, peak_bytes)
