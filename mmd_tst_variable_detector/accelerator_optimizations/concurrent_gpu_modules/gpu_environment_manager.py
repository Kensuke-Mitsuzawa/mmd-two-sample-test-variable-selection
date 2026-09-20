import logging
import os
import shutil
import subprocess
import typing as ty
import torch

from .exceptions import IncompatibleGpuArchitectureError

logger = logging.getLogger(__name__)


def assert_device_compatibility(device_id: int = 0) -> None:
    """Verify that the target CUDA device is fully supported by the installed PyTorch build.

    Raises
    ------
    RuntimeError
        If CUDA is not available on the host.
    IncompatibleGpuArchitectureError
        If the device's compute capability is not supported by the installed PyTorch build
        or if kernel execution fails due to architecture incompatibility.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")

    device_count = torch.cuda.device_count()
    if device_id < 0 or device_id >= device_count:
        raise ValueError(
            f"Requested device_id {device_id} is out of range. Available devices: {device_count}."
        )

    device_name = torch.cuda.get_device_name(device_id)
    major, minor = torch.cuda.get_device_capability(device_id)
    device_cc = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()

    # If architecture is explicitly missing from PyTorch's compiled arch list
    compatible = (device_cc in arch_list) or (f"sm_{major}0" in arch_list)
    if arch_list and not compatible:
        raise IncompatibleGpuArchitectureError(
            device_name=device_name,
            device_capability=device_cc,
            supported_architectures=arch_list,
        )

    # Canary check: execute a tiny kernel to ensure CUDA driver and runtime execute without error
    try:
        canary = torch.zeros(1, device=f"cuda:{device_id}") + 1
        _ = canary.item()
    except Exception as exc:
        raise IncompatibleGpuArchitectureError(
            device_name=device_name,
            device_capability=device_cc,
            supported_architectures=arch_list,
            message=str(exc),
        ) from exc


class GpuEnvironmentManager(object):
    """Manages GPU environment detection, device verification, and NVIDIA MPS lifecycle."""

    def __init__(self, enable_mps: bool = True):
        self.enable_mps = enable_mps
        self.mps_active = False

    @staticmethod
    def get_gpu_count() -> int:
        """Return the number of available CUDA devices."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()

    @staticmethod
    def verify_all_gpus() -> None:
        """Verify all available GPUs on the host for PyTorch compatibility."""
        count = GpuEnvironmentManager.get_gpu_count()
        if count == 0:
            raise RuntimeError("No CUDA devices detected.")
        for dev_id in range(count):
            assert_device_compatibility(dev_id)

    def start_mps(self) -> bool:
        """Attempt to start the NVIDIA MPS control daemon.

        Returns
        -------
        bool
            True if MPS started successfully, False if skipped or fallen back.
        """
        if not self.enable_mps:
            logger.info("NVIDIA MPS is disabled by configuration.")
            return False

        mps_binary = shutil.which("nvidia-cuda-mps-control")
        if not mps_binary:
            logger.warning(
                "nvidia-cuda-mps-control not found on PATH. Falling back to CUDA time-slicing."
            )
            self.mps_active = False
            return False

        try:
            # Set pipe directories to a writable user directory if not already configured
            os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
            os.environ.setdefault("CUDA_MPS_LOG_DIRECTORY", "/tmp/nvidia-log")
            os.makedirs(os.environ["CUDA_MPS_PIPE_DIRECTORY"], exist_ok=True)
            os.makedirs(os.environ["CUDA_MPS_LOG_DIRECTORY"], exist_ok=True)

            cmd = [mps_binary, "-d"]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                logger.info("NVIDIA MPS daemon started successfully.")
                self.mps_active = True
                return True
            else:
                logger.warning(
                    f"Failed to start NVIDIA MPS (exit code {res.returncode}): {res.stderr}. "
                    "Falling back to standard CUDA time-slicing."
                )
                self.mps_active = False
                return False
        except Exception as exc:
            logger.warning(
                f"Exception starting NVIDIA MPS: {exc}. Falling back to standard CUDA time-slicing."
            )
            self.mps_active = False
            return False

    def stop_mps(self) -> None:
        """Stop the NVIDIA MPS control daemon if running."""
        if not self.mps_active:
            return

        mps_binary = shutil.which("nvidia-cuda-mps-control")
        if not mps_binary:
            self.mps_active = False
            return

        try:
            proc = subprocess.Popen(
                [mps_binary],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            proc.communicate(input="quit\n", timeout=5)
            logger.info("NVIDIA MPS daemon stopped.")
        except Exception as exc:
            logger.warning(f"Failed to cleanly stop NVIDIA MPS daemon: {exc}")
        finally:
            self.mps_active = False

    def __enter__(self) -> "GpuEnvironmentManager":
        self.start_mps()
        return self

    def __exit__(
        self,
        exc_type: ty.Optional[ty.Type[BaseException]],
        exc_val: ty.Optional[BaseException],
        exc_tb: ty.Any,
    ) -> None:
        self.stop_mps()
