class IncompatibleGpuArchitectureError(RuntimeError):
    """Exception raised when the target GPU hardware compute capability
    is not supported by the installed PyTorch binary.
    """

    def __init__(
        self,
        device_name: str,
        device_capability: str,
        supported_architectures: list[str],
        message: str = "",
    ):
        self.device_name = device_name
        self.device_capability = device_capability
        self.supported_architectures = supported_architectures

        detail = (
            f"GPU device '{device_name}' has compute capability {device_capability}, "
            f"but the installed PyTorch build only supports architectures: {supported_architectures}. "
            "Execution is halted to prevent CUDA runtime crashes. "
            "Please install a PyTorch build compatible with this GPU architecture (e.g. PyTorch with compatible CUDA build) "
            "or use a supported GPU device."
        )
        if message:
            detail = f"{detail} Additional info: {message}"
        super().__init__(detail)
